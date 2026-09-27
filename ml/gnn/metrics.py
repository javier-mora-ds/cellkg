"""Evaluation harness (plan §5.H) — numpy only, no torch.

Every metric is computed from a score matrix ``S`` [n, 22], the positives to evaluate ``Y_pos``,
the known-but-not-evaluated positives ``Y_mask`` and a candidate mask (C22 or C14). A gene-weight
matrix ``W`` [B, n] turns the same code into the gene-level bootstrap (row b = multiplicities of
resample b; a single row of ones = the point estimate).
"""
from __future__ import annotations

import warnings

import numpy as np

RANK_OFFSET = {"optimistic": 0.0, "pessimistic": 1.0, "realistic": 0.5}


# ---------------------------------------------------------------- filtered tail-side ranks
def positive_ranks(
    S: np.ndarray, Y_pos: np.ndarray, Y_mask: np.ndarray, cand: np.ndarray, ties: str = "realistic"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filtered rank of every evaluated positive: other true labels and non-candidates excluded."""
    rows, cols = np.nonzero(Y_pos)
    if not np.isfinite(S[rows]).all():
        raise ValueError("score matrix has non-finite values on evaluated rows")
    s = S[rows]
    st = s[np.arange(len(rows)), cols][:, None]
    excl = Y_pos[rows] | Y_mask[rows] | ~cand[None, :]
    excl[np.arange(len(rows)), cols] = False
    better = ((s > st) & ~excl).sum(1)
    tied = ((s == st) & ~excl).sum(1) - 1
    return rows, cols, 1.0 + better + RANK_OFFSET[ties] * tied


def lrap_pairs(
    S: np.ndarray,
    Y_pos: np.ndarray,
    Y_mask: np.ndarray,
    cand: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Per-positive LRAP term |{t' in Y_g: rank t' <= rank t}| / rank t (unfiltered among candidates)."""
    s = S[rows]
    st = s[np.arange(len(rows)), cols][:, None]
    avail = ~Y_mask[rows] & cand[None, :]
    pos = Y_pos[rows] & avail
    rank = ((s > st) & avail).sum(1) + 0.5 * (((s == st) & avail).sum(1) - 1) + 1
    above = ((s > st) & pos).sum(1) + 0.5 * (((s == st) & pos).sum(1) - 1) + 1
    return above / rank


# ---------------------------------------------------------------- column-wise curves
def ap_auc(y: np.ndarray, s: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted average precision and AUROC for one column, for every row of W [B, n].

    Step-wise AP (sklearn definition) and trapezoidal AUROC over distinct score thresholds.
    """
    order = np.argsort(-s, kind="mergesort")
    y, s, W = y[order].astype(float), s[order], W[:, order]
    ends = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(W * y, axis=1)[:, ends]
    fp = np.cumsum(W * (1 - y), axis=1)[:, ends]
    P, N = tp[:, -1:], fp[:, -1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = tp / P
        d_rec = np.diff(np.hstack([np.zeros_like(P), rec]), axis=1)
        ap = (d_rec * prec).sum(1)
        tpr, fpr = rec, fp / N
        tpr0 = np.hstack([np.zeros_like(P), tpr])
        fpr0 = np.hstack([np.zeros_like(N), fpr])
        auc = (np.diff(fpr0, axis=1) * (tpr0[:, 1:] + tpr0[:, :-1]) / 2).sum(1)
    ap = np.where(P[:, 0] > 0, ap, np.nan)
    auc = np.where((P[:, 0] > 0) & (N[:, 0] > 0), auc, np.nan)
    return ap, auc


# ---------------------------------------------------------------- thresholds (validation only)
def _best_cut(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order].astype(float), s[order]
    ends = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[ends]
    fp = np.cumsum(1 - y)[ends]
    P = y.sum()
    f1 = 2 * tp / np.maximum(2 * tp + fp + (P - tp), 1e-12)
    j = int(np.argmax(f1))
    return float(s[ends[j]]), float(f1[j])


def fit_thresholds(
    S: np.ndarray, Y: np.ndarray, M: np.ndarray, supported: np.ndarray, min_pos: int = 5
) -> np.ndarray:
    """theta_o = argmax validation F1 per class; classes with < min_pos positives share the
    global threshold that maximizes validation micro-F1; unsupported columns never fire."""
    theta = np.full(S.shape[1], np.inf)
    valid = ~M
    pool_y = np.hstack([Y[valid[:, o], o] for o in np.flatnonzero(supported)])
    pool_s = np.hstack([S[valid[:, o], o] for o in np.flatnonzero(supported)])
    glob = _best_cut(pool_y, pool_s)[0] if pool_y.any() else np.inf
    for o in np.flatnonzero(supported):
        y, s = Y[valid[:, o], o], S[valid[:, o], o]
        theta[o] = _best_cut(y, s)[0] if y.sum() >= min_pos else glob
    return theta


# ---------------------------------------------------------------- the harness
class Harness:
    """Holds one evaluation set; `evaluate` returns every metric as arrays over the rows of W."""

    def __init__(
        self,
        Y_pos: np.ndarray,
        Y_mask: np.ndarray,
        codes: tuple[str, ...],
        supported: np.ndarray,
        rare_tail: list[str],
        ultra_rare: list[str],
        ks=(1, 3, 10),
        alphas=(0.0, 0.5, 1.0),
        ties: str = "realistic",
    ):
        self.Y_pos, self.Y_mask = Y_pos.astype(bool), Y_mask.astype(bool)
        self.codes, self.n_org = codes, len(codes)
        self.c22 = np.ones(self.n_org, dtype=bool)
        self.c14 = supported.astype(bool)
        self.R = np.isin(codes, rare_tail)
        self.U = np.isin(codes, ultra_rare)
        self.ks, self.alphas, self.ties = tuple(ks), tuple(alphas), ties

    # rank-based -----------------------------------------------------------
    def _rank_block(self, S, W, cand, tag, out) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows, cols, rk = positive_ranks(S, self.Y_pos, self.Y_mask, cand, self.ties)
        onehot = np.zeros((len(cols), self.n_org))
        onehot[np.arange(len(cols)), cols] = 1.0
        PW = W[:, rows]  # [B, P]
        m = PW @ onehot  # weighted support per class
        rr = 1.0 / rk
        with np.errstate(invalid="ignore", divide="ignore"):
            mrr_o = (PW * rr) @ onehot / m
            hits_o = {k: (PW * (rk <= k)) @ onehot / m for k in self.ks}
        present = m > 0
        for a in self.alphas:
            wa = np.where(present, np.power(np.where(present, m, 1.0), 1.0 - a), 0.0)
            out[f"MRR_a{a:g}@{tag}"] = np.nansum(wa * np.nan_to_num(mrr_o), 1) / wa.sum(1)
            for k in self.ks:
                out[f"Hits@{k}_a{a:g}@{tag}"] = np.nansum(
                    wa * np.nan_to_num(hits_o[k]), 1
                ) / wa.sum(1)
        if tag == "C14":
            out["MRR_o"] = mrr_o
            out["Hits@1_o"] = hits_o[1]
            out["support_o"] = m
            for nm, sel in (("R", self.R), ("U", self.U)):
                pr = present[:, sel]
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[f"{nm}_macroMRR"] = np.nansum(np.where(pr, mrr_o[:, sel], 0), 1) / pr.sum(1)
                    out[f"{nm}_microMRR"] = (PW * rr) @ onehot[:, sel].sum(1) / m[:, sel].sum(1)
        return rows, cols, rk

    # evaluate -------------------------------------------------------------
    def evaluate(
        self, S: np.ndarray, pred: np.ndarray | None = None, W: np.ndarray | None = None
    ) -> dict:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # empty rare-class slices -> NaN by design
            return self._evaluate(S, pred, W)

    def _evaluate(self, S: np.ndarray, pred: np.ndarray | None, W: np.ndarray | None) -> dict:
        n = S.shape[0]
        W = np.ones((1, n)) if W is None else W
        out: dict[str, np.ndarray] = {}
        self._rank_block(S, W, self.c22, "C22", out)
        rows, cols, _ = self._rank_block(S, W, self.c14, "C14", out)
        # LRAP (C14), gene-level mean then weighted over genes with positives
        v = lrap_pairs(S, self.Y_pos, self.Y_mask, self.c14, rows, cols)
        g_lrap = np.bincount(rows, weights=v, minlength=n) / np.maximum(
            np.bincount(rows, minlength=n), 1
        )
        has = self.Y_pos.any(1)
        out["LRAP"] = (W[:, has] @ g_lrap[has]) / W[:, has].sum(1)
        # column-wise AP / AUROC over the supported organelles
        ap = np.full((W.shape[0], self.n_org), np.nan)
        auc = np.full((W.shape[0], self.n_org), np.nan)
        for o in np.flatnonzero(self.c14):
            ok = ~self.Y_mask[:, o]
            ap[:, o], auc[:, o] = ap_auc(self.Y_pos[ok, o], S[ok, o], W[:, ok])
        out["AP_o"], out["AUROC_o"] = ap, auc
        with np.errstate(invalid="ignore"):
            out["mAP"] = np.nanmean(ap[:, self.c14], 1)
            out["macroAUROC"] = np.nanmean(auc[:, self.c14], 1)
            out["R_mAP"] = np.nanmean(ap[:, self.R], 1)
            out["U_mAP"] = np.nanmean(ap[:, self.U], 1)
        if pred is not None:
            ok = ~self.Y_mask
            tp = W @ (pred & self.Y_pos & ok).astype(float)
            fp = W @ (pred & ~self.Y_pos & ok).astype(float)
            fn = W @ (~pred & self.Y_pos & ok).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                f1 = np.where(tp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
                out["F1_o"] = f1
                out["macroF1"] = np.nanmean(f1[:, self.c14], 1)
                sel = self.c14
                out["microF1"] = (
                    2
                    * tp[:, sel].sum(1)
                    / (2 * tp[:, sel].sum(1) + fp[:, sel].sum(1) + fn[:, sel].sum(1))
                )
        return out


def point(metrics: dict) -> dict:
    """Collapse a single-row evaluation to plain floats / per-class lists."""
    out = {}
    for k, v in metrics.items():
        v = np.asarray(v)
        out[k] = v[0].tolist() if v.ndim == 2 else float(v[0])
    return out


def bootstrap_weights(n: int, draws: int, seed: int) -> np.ndarray:
    """Multiplicity matrix [draws, n] of a gene-level bootstrap (same seed -> paired across models)."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(draws)]).astype(
        float
    )


def skill(m: np.ndarray, m_prior: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (m - m_prior) / (1.0 - m_prior)
