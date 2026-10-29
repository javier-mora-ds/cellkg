"""Comparison tables (T1-T6, panel) and figures (F1-F5) -> reports/gnn/ (plan §5.I)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from gnn.common import resolve_args  # noqa: E402
from gnn.common import (
    OrganelleVocab,
    StageLog,
    base_parser,
    located_in_edges,
    out_dir,
    update_manifest,
)
from gnn.splits import P1_DIR, split_root  # noqa: E402

NAMES = {
    "prior": "Prior (popularity)",
    "ppi_vote": "PPI neighbour vote",
    "label_prop": "Label propagation",
    "TransE": "TransE (re-trained)",
    "RotatE": "RotatE (re-trained)",
    "mlp_bce": "MLP-BCE",
    "mlp_la": "MLP-LA",
    "mlp_bce_posthoc": "MLP-BCE + post-hoc LA",
    "sage_bce": "SAGE-BCE",
    "sage_la": "SAGE-LA",
    "sage_bce_posthoc": "SAGE-BCE + post-hoc LA",
    "sage_cb": "SAGE-CB",
    "sage_asl": "SAGE-ASL",
    "sage_la_rho0": "SAGE-LA, rho = 0 (S4)",
    "mlp_bce_xperm": "MLP-BCE (permuted X)",
    "mlp_bce_xperm_posthoc": "MLP-BCE (permuted X) + post-hoc LA",
    "sage_bce_xperm": "SAGE-BCE (permuted X)",
    "sage_bce_xperm_posthoc": "SAGE-BCE (permuted X) + post-hoc LA",
}
ORDER = list(NAMES)
PUBLISHED = [
    ("Random (published)", 0.2369, None, None),
    ("Degree (published)", 0.5936, 0.4400, 0.6600),
    ("TransE (published)", 0.2941, 0.1350, 0.2750),
    ("RotatE (published)", 0.4917, 0.3400, 0.5550),
]
RELIABILITY = {"Enhanced": 1.0, "Supported": 0.8, "Approved": 0.6, "Uncertain": 0.3}
# validated categorical order (light surface); identity also carried by marker shape and direct labels
C = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#898781", "#e1e0d9", "#fcfcfb"
MODEL_STYLE = {
    "prior": (MUTED, "s"),
    "sage_la": (C["blue"], "P"),
    "RotatE": (C["orange"], "D"),
    "sage_bce": (C["aqua"], "P"),
    "TransE": (C["yellow"], "D"),
    "mlp_bce": (C["magenta"], "o"),
    "label_prop": (C["green"], "^"),
    "mlp_la": (C["violet"], "o"),
    "ppi_vote": (C["red"], "v"),
}
LABEL_OFFSET = {"prior": (6, 6), "RotatE": (8, 4), "sage_bce": (8, -12), "sage_la": (-8, 8)}
# factorial 2 x 2 cells: T5 crosses the encoder with the loss; T5b crosses the encoder with the attributes
# (real X vs row-permuted X, the topology-only control); the second factor is always b, d vs a, c
FACTORIAL_LOSS = (("a", "mlp_bce"), ("b", "mlp_la"), ("c", "sage_bce"), ("d", "sage_la"))
FACTORIAL_XPERM = (
    ("a", "mlp_bce_xperm"),
    ("b", "mlp_bce"),
    ("c", "sage_bce_xperm"),
    ("d", "sage_bce"),
)
XPERM_PAIRS = (
    ("sage_bce", "sage_bce_xperm"),
    ("sage_bce_xperm", "mlp_bce_xperm"),
    ("sage_bce_xperm", "prior"),
    ("sage_bce_xperm", "label_prop"),
    ("mlp_bce_xperm", "prior"),
    ("mlp_bce", "mlp_bce_xperm"),
)
KEYS_T2 = [
    ("MRR_a0@C14", "micro-MRR"),
    ("MRR_a0.5@C14", "MRR a=0.5"),
    ("MRR_a1@C14", "macro-MRR"),
    ("Hits@1_a0@C22", "H@1 C22"),
    ("Hits@1_a0@C14", "H@1 C14"),
    ("Hits@3_a0@C22", "H@3 C22"),
    ("Hits@3_a0@C14", "H@3 C14"),
    ("mAP", "mAP"),
    ("macroAUROC", "macro-AUROC"),
    ("LRAP", "LRAP"),
    ("macroF1", "macro-F1"),
    ("R_macroMRR", "R macro-MRR"),
    ("R_mAP", "R mAP"),
    ("U_mAP", "U mAP"),
    ("coverage_test", "coverage"),
]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


class Results:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.root = out_dir()
        self.summary = pd.read_csv(self.root / "metrics" / "summary.csv")
        self.seed = cfg["seed"]
        self.vocab = OrganelleVocab.load()
        self.protocol = json.loads((split_root(P1_DIR) / "split_manifest.json").read_text())[
            "gate_a"
        ]["protocol"]

    def agg(self, eval_set: str, regime: str | None = None) -> pd.DataFrame:
        d = self.summary[self.summary.eval_set == eval_set]
        if regime:
            d = d[d.regime == regime]
        num = d.select_dtypes("number").columns.drop("seed")
        g = d.groupby(["model", "regime"])[list(num)]
        m, s, n = g.mean(), g.std(ddof=1), g.size()
        m.columns = pd.MultiIndex.from_product([["mean"], m.columns])
        s.columns = pd.MultiIndex.from_product([["sd"], s.columns])
        cnt = n.to_frame()
        cnt.columns = pd.MultiIndex.from_tuples([("n", "seeds")])
        return pd.concat([m, s, cnt], axis=1)

    def tag(self, model: str, regime: str, eval_set: str) -> str:
        if eval_set == "p2_pooled":
            return f"{model}_p2_pooled_seed{self.seed}"
        return f"{model}_{self.protocol}_{regime}_{eval_set}_seed{self.seed}"

    def boot(self, model: str, regime: str, eval_set: str) -> dict | None:
        p = self.root / "metrics" / f"bootstrap_{self.tag(model, regime, eval_set)}.npz"
        return dict(np.load(p)) if p.exists() else None

    def point(self, model: str, regime: str, eval_set: str) -> dict | None:
        p = self.root / "metrics" / f"{self.tag(model, regime, eval_set)}.json"
        return json.loads(p.read_text()) if p.exists() else None


def ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if not len(x):
        return (np.nan, np.nan)
    a = (1 - level) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


def fmt(m: float, s: float | None = None, nd: int = 3) -> str:
    if m is None or not np.isfinite(m):
        return "—"
    return f"{m:.{nd}f}" + (f" ± {s:.{nd}f}" if s is not None and np.isfinite(s) and s > 0 else "")


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|",
    ]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _models_present(agg: pd.DataFrame) -> list[str]:
    ms = {m for m, _ in agg.index}
    return [m for m in ORDER if m in ms]


# ---------------------------------------------------------------- tables
def table_t1(R: Results) -> pd.DataFrame:
    rows = [
        {
            "Method": n,
            "Regime": "—",
            "MRR C22": fmt(a, nd=4),
            "Hits@1 C22": fmt(b, nd=4) if b else "—",
            "Hits@3 C22": fmt(c, nd=4) if c else "—",
            "MRR C14": "—",
            "Hits@1 C14": "—",
            "Hits@3 C14": "—",
        }
        for n, a, b, c in PUBLISHED
    ]
    agg = R.agg("p1_sub200")
    for m in _models_present(agg):
        for reg in ("strict", "full"):
            if (m, reg) not in agg.index:
                continue
            r = agg.loc[(m, reg)]
            g = lambda k: fmt(r[("mean", k)], r[("sd", k)], 4)  # noqa: E731
            rows.append(
                {
                    "Method": NAMES[m],
                    "Regime": reg,
                    "MRR C22": g("MRR_a0@C22"),
                    "Hits@1 C22": g("Hits@1_a0@C22"),
                    "Hits@3 C22": g("Hits@3_a0@C22"),
                    "MRR C14": g("MRR_a0@C14"),
                    "Hits@1 C14": g("Hits@1_a0@C14"),
                    "Hits@3 C14": g("Hits@3_a0@C14"),
                }
            )
    return pd.DataFrame(rows)


def table_metrics(
    R: Results, eval_set: str, regime: str | None, with_ci: bool = True
) -> pd.DataFrame:
    agg = R.agg(eval_set, regime)
    rows = []
    for m in _models_present(agg):
        for reg in sorted({r for mm, r in agg.index if mm == m}):
            r = agg.loc[(m, reg)]
            row = {"Method": NAMES[m], "seeds": int(r[("n", "seeds")])}
            for k, lab in KEYS_T2:
                if ("mean", k) in r.index:
                    row[lab] = fmt(r[("mean", k)], r[("sd", k)])
            if with_ci:
                b = R.boot(m, reg, eval_set)
                if b is not None:
                    for k, lab in (
                        ("MRR_a0@C14", "micro CI"),
                        ("MRR_a1@C14", "macro CI"),
                        ("mAP", "mAP CI"),
                    ):
                        lo, hi = ci(b[k])
                        row[lab] = f"[{lo:.3f}, {hi:.3f}]"
            rows.append(row)
    return pd.DataFrame(rows)


def table_t3(R: Results, eval_set: str, regime: str, n_train: np.ndarray) -> pd.DataFrame:
    loc = located_in_edges(R.vocab)
    rel = loc.assign(w=loc["reliability"].map(RELIABILITY)).groupby("code")["w"].mean()
    pts = {m: R.point(m, regime, eval_set) for m in ("prior", "RotatE", "sage_bce", "sage_la")}
    b = R.boot("sage_la", regime, eval_set)
    bp = R.boot("prior", regime, eval_set)
    sup = np.array(pts["prior"]["support_o"])
    ap_prior = np.array(pts["prior"]["AP_o"], dtype=float)
    rows = []
    for o in np.argsort(-np.nan_to_num(sup), kind="mergesort"):
        if not sup[o] > 0:
            continue
        c = R.vocab.codes[o]
        g = lambda m, k: pts[m][k][o] if pts[m] else np.nan  # noqa: E731
        ap_la = g("sage_la", "AP_o")
        sk = (ap_la - ap_prior[o]) / (1 - ap_prior[o])
        if b is not None and bp is not None:
            skb = (b["AP_o"][:, o] - bp["AP_o"][:, o]) / (1 - bp["AP_o"][:, o])
            lo, hi = ci(skb)
            ci_s = f"[{lo:.3f}, {hi:.3f}]"
        else:
            ci_s = "—"
        rows.append(
            {
                "Code": c,
                "n_train": int(n_train[o]),
                "m_test": int(sup[o]),
                "Prior MRR_o": fmt(g("prior", "MRR_o")),
                "RotatE MRR_o": fmt(g("RotatE", "MRR_o")),
                "SAGE-BCE MRR_o": fmt(g("sage_bce", "MRR_o")),
                "SAGE-LA MRR_o": fmt(g("sage_la", "MRR_o")),
                "SAGE-LA AP_o (prior = prevalence)": f"{fmt(ap_la)} ({fmt(ap_prior[o])})",
                "SAGE-LA AUROC_o": fmt(g("sage_la", "AUROC_o")),
                "AP skill SAGE-LA": fmt(sk),
                "95 % CI": ci_s,
                "Mean reliability": fmt(rel.get(c, np.nan), nd=2),
            }
        )
    return pd.DataFrame(rows)


def factorial(
    R: Results,
    eval_set: str,
    regime: str,
    cells_of: tuple = FACTORIAL_LOSS,
    factor2: str = "loss",
) -> pd.DataFrame | None:
    """2 x 2 contrasts (graph, `factor2`, interaction) over the cells a to d named by `cells_of`."""
    cells = {k: R.boot(m, regime, eval_set) for k, m in cells_of}
    pts = {k: R.point(m, regime, eval_set) for k, m in cells_of}
    if any(v is None for v in cells.values()) or any(v is None for v in pts.values()):
        return None
    rows = []
    for k, lab in (("MRR_a1@C14", "macro-MRR"), ("mAP", "mAP"), ("MRR_a0@C14", "micro-MRR")):
        a, b, c, d = (cells[x][k] for x in "abcd")
        pa, pb, pc, pd_ = (pts[x][k] for x in "abcd")
        eff = {
            "graph": ((c + d) / 2 - (a + b) / 2, (pc + pd_) / 2 - (pa + pb) / 2),
            factor2: ((b + d) / 2 - (a + c) / 2, (pb + pd_) / 2 - (pa + pc) / 2),
            "interaction": ((d - c) - (b - a), (pd_ - pc) - (pb - pa)),
        }
        row = {"Metric": lab, **{f"{x} {NAMES[m]}": fmt(pts[x][k]) for x, m in cells_of}}
        for nm, (boot, pt) in eff.items():
            lo, hi = ci(boot)
            row[f"{nm} effect [95 % CI]"] = f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def paired(R: Results, eval_set: str, regime: str, a: str, b: str) -> list[dict]:
    ba, bb = R.boot(a, regime, eval_set), R.boot(b, regime, eval_set)
    pa, pb = R.point(a, regime, eval_set), R.point(b, regime, eval_set)
    if ba is None or bb is None:
        return []
    out = []
    for k, lab in (
        ("MRR_a0@C14", "micro-MRR C14"),
        ("MRR_a0@C22", "micro-MRR C22"),
        ("MRR_a1@C14", "macro-MRR"),
        ("mAP", "mAP"),
        ("R_macroMRR", "R macro-MRR"),
        ("U_mAP", "U mAP"),
    ):
        d = ba[k] - bb[k]
        lo, hi = ci(d)
        out.append(
            {
                "Comparison": f"{NAMES[a]} − {NAMES[b]}",
                "Eval": f"{eval_set}/{regime}",
                "Metric": lab,
                "Difference": f"{pa[k] - pb[k]:+.4f}",
                "95 % CI": f"[{lo:+.4f}, {hi:+.4f}]",
                "P(diff > 0)": f"{np.mean(d[np.isfinite(d)] > 0):.3f}",
            }
        )
    return out


# ---------------------------------------------------------------- figures
def fig_f1(R: Results, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    Hc = sum(1 / k for k in range(1, 15)) / 14
    for ax, (es, reg, title) in zip(
        axes,
        (
            ("p1_full", "full", "P1-full test edges"),
            ("p2_pooled", "pooled", "P2 pooled out-of-fold"),
        ),
    ):
        _style(ax)
        pr = R.point("prior", "full" if es == "p1_full" else "pooled", es)
        for m, (col, mk) in MODEL_STYLE.items():
            p, b = R.point(m, reg, es), R.boot(m, reg, es)
            if p is None:
                continue
            x, y = p["MRR_a0@C14"], p["MRR_a1@C14"]
            xe = (
                np.array([[x - ci(b["MRR_a0@C14"])[0]], [ci(b["MRR_a0@C14"])[1] - x]])
                if b
                else None
            )
            ye = (
                np.array([[y - ci(b["MRR_a1@C14"])[0]], [ci(b["MRR_a1@C14"])[1] - y]])
                if b
                else None
            )
            ax.errorbar(
                x,
                y,
                xerr=xe,
                yerr=ye,
                fmt=mk,
                color=col,
                ms=8,
                mec=SURFACE,
                mew=1.5,
                elinewidth=1.2,
                capsize=0,
                label=NAMES[m],
            )
            if m in LABEL_OFFSET:  # selective direct labels; the legend carries the rest
                dx, dy = LABEL_OFFSET[m]
                ax.annotate(
                    NAMES[m].replace(" (re-trained)", "").replace(" (popularity)", ""),
                    (x, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8,
                    color=INK,
                    ha="right" if dx < 0 else "left",
                )
        if pr:
            ax.axhline(pr["MRR_a1@C14"], color=MUTED, lw=1, ls="--")
            ax.axvline(pr["MRR_a0@C14"], color=MUTED, lw=1, ls="--")
        ax.plot(Hc, Hc, marker="x", color=MUTED, ms=8)
        ax.annotate(
            "random (H14/14)",
            (Hc, Hc),
            xytext=(5, -10),
            textcoords="offset points",
            fontsize=7,
            color=MUTED,
        )
        ax.set_xlabel("micro-MRR (C14; rewards frequent organelles)", fontsize=9, color=INK)
        ax.set_ylabel("macro-MRR (C14; every organelle weighs 1)", fontsize=9, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center", ncol=5)
    fig.suptitle(
        "F1 — micro vs macro MRR (dashed: popularity prior; 95 % gene-bootstrap intervals)",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_f2(R: Results, path) -> None:
    es, reg = "p2_pooled", "pooled"
    pr = R.point("prior", reg, es)
    sup = np.array(pr["support_o"], dtype=float)
    idx = [o for o in np.argsort(-np.nan_to_num(sup), kind="mergesort") if sup[o] > 0]
    models = [(m, MODEL_STYLE[m][0]) for m in ("sage_la", "RotatE", "sage_bce")]
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=SURFACE)
    _style(ax)
    w = 0.26
    bp = R.boot("prior", reg, es)
    for j, (m, col) in enumerate(models):
        p, b = R.point(m, reg, es), R.boot(m, reg, es)
        if p is None:
            continue
        ap, app = np.array(p["AP_o"], dtype=float), np.array(pr["AP_o"], dtype=float)
        sk = (ap - app) / (1 - app)
        xs = np.arange(len(idx)) + (j - 1) * w
        err = None
        if b is not None and bp is not None:
            skb = (b["AP_o"] - bp["AP_o"]) / (1 - bp["AP_o"])
            lo = np.array([ci(skb[:, o])[0] for o in idx])
            hi = np.array([ci(skb[:, o])[1] for o in idx])
            err = np.vstack([sk[idx] - lo, hi - sk[idx]])
        ax.bar(
            xs,
            sk[idx],
            width=w - 0.03,
            color=col,
            label=NAMES[m],
            yerr=err,
            ecolor=INK,
            error_kw={"elinewidth": 0.8},
        )
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(np.arange(len(idx)))
    ax.set_xticklabels([f"{R.vocab.codes[o]}\nm={int(sup[o])}" for o in idx], fontsize=7)
    ax.set_ylabel("AP skill vs prior  (AP − prev) / (1 − prev)", fontsize=9, color=INK)
    ax.set_title(
        "F2 — per-organelle skill score (P2 pooled; organelles sorted by support; 95 % CI)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_f3(R: Results, path) -> bool:
    from gnn.graph import NodeIndex
    from gnn.splits import make_task
    from sklearn.manifold import TSNE

    e = R.root / "models" / f"sage_la_{R.protocol}_full_seed{R.seed}_embedding.npz"
    if not e.exists():
        return False
    h = np.load(e)["h"]
    index = NodeIndex.load()
    task = make_task("p1", "full", index, R.vocab)
    Y = task.labels("test", len(index.genes), len(R.vocab))
    rows = task.test_genes[Y[task.test_genes].sum(1) == 1]
    gold = Y[rows].argmax(1)
    Z = TSNE(n_components=2, random_state=R.seed, init="pca", perplexity=30).fit_transform(h[rows])
    codes = np.array(R.vocab.codes)
    n_train = np.bincount(task.loc_train[:, 1], minlength=len(R.vocab))
    prank = np.empty(len(R.vocab))
    prank[np.argsort(-n_train, kind="mergesort")] = np.arange(1, len(R.vocab) + 1)
    groups = [
        ("NUC", C["blue"]),
        ("CTP", C["orange"]),
        ("PMB", C["aqua"]),
        ("TRV", C["yellow"]),
        ("MTC", C["magenta"]),
    ]
    rare = {"CTK", "ERT", "CTS", "NVL", "CRM", "PRX", "LYS"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    lab = codes[gold]
    other = ~np.isin(lab, [g for g, _ in groups]) & ~np.isin(lab, list(rare))
    ax.scatter(*Z[other].T, s=6, color=GRID, label="other (NCL, GLG)")
    for g, col in groups:
        s = lab == g
        ax.scatter(*Z[s].T, s=6, color=col, label=g)
    s = np.isin(lab, list(rare))
    ax.scatter(
        *Z[s].T,
        s=14,
        color=C["violet"],
        marker="^",
        label="rare tail R",
        edgecolors=SURFACE,
        linewidths=0.4,
    )
    ax.legend(fontsize=7, frameon=False, markerscale=2, loc="best")
    ax.set_title(
        "coloured by gold organelle (single-label test genes)", fontsize=9, color=INK, loc="left"
    )
    ax = axes[1]
    _style(ax)
    sc = ax.scatter(*Z.T, s=6, c=prank[gold], cmap="Blues_r", vmin=1, vmax=14)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("prior rank of the gold organelle (1 = NUC)", fontsize=8)
    ax.set_title("coloured by prior rank of the gold label", fontsize=9, color=INK, loc="left")
    for a in axes:
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle(
        f"F3 — t-SNE of SAGE-LA gene embeddings h_g (P1-full, {len(rows)} test genes)",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return True


def fig_f4(R: Results, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE, sharey=False)
    for ax, m in zip(axes, ("sage_bce", "sage_la")):
        _style(ax)
        f = R.root / "curves" / f"{m}_{R.protocol}_full_seed{R.seed}.csv"
        if not f.exists():
            continue
        c = pd.read_csv(f)
        for k, lab, col in (
            ("val_micro_mrr", "val micro-MRR", C["orange"]),
            ("val_macro_mrr", "val macro-MRR", C["aqua"]),
            ("val_map_elig", "val mAP (criterion)", C["blue"]),
        ):
            ax.plot(c["epoch"], c[k], color=col, lw=2, label=lab)
            j = int(c[k].idxmax())
            ax.plot(c["epoch"][j], c[k][j], marker="o", color=col, ms=8, mec=SURFACE, mew=1.5)
        ax.set_xlabel("epoch", fontsize=9)
        ax.set_title(
            f"{NAMES[m]} (P1-full); dots = best epoch per criterion",
            fontsize=9,
            color=INK,
            loc="left",
        )
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("F4 — training curves", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_f5(R: Results, path) -> None:
    from gnn.common import ScoreMatrix
    from gnn.evaluate import p2_eval_set
    from gnn.graph import NodeIndex
    from gnn.splits import make_task
    from sklearn.metrics import precision_recall_curve

    index = NodeIndex.load()
    folds = [f"fold_{k}" for k in range(1, R.cfg["p2"]["n_folds"] + 1)]
    es = p2_eval_set([make_task("p2", r, index, R.vocab) for r in folds], index, R.vocab)
    S = np.zeros(es.Y_pos.shape, dtype=np.float32)
    for k, r in enumerate(folds, start=1):
        p = R.root / "scores" / f"sage_la_p2_{r}_seed{R.seed}.npz"
        if not p.exists():
            return
        sel = es.fold == k
        S[sel] = ScoreMatrix.load(p).S[es.rows[sel]]
    codes = list(R.vocab.codes)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    pal = [C[k] for k in ("blue", "orange", "aqua", "yellow", "magenta", "green", "violet")]
    for ax, (title, group) in zip(
        axes,
        (
            ("ultra-rare U", R.cfg["metrics"]["ultra_rare"]),
            ("rare tail R", R.cfg["metrics"]["rare_tail"]),
        ),
    ):
        _style(ax)
        for c, col in zip(group, pal):
            o = codes.index(c)
            y = es.Y_pos[:, o]
            pr, rc, _ = precision_recall_curve(y, S[:, o])
            ax.step(rc, pr, where="post", color=col, lw=2, label=f"{c} (n={int(y.sum())})")
            ax.axhline(y.mean(), color=col, lw=0.8, ls="--")
        ax.set_xlabel("recall", fontsize=9)
        ax.set_ylabel("precision", fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{title}; dashed = prior (prevalence)", fontsize=9, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(
        "F5 — precision–recall of SAGE-LA on pooled P2 (seed 20260702)",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = base_parser("Comparison tables and figures -> reports/gnn/.")
    args = ap.parse_args()
    cfg = resolve_args(args)
    log = StageLog("compare")
    R = Results(cfg)
    root = R.root
    fdir = root / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    from gnn.splits import make_task

    t1 = make_task("p1", "full")
    n_train_p1 = np.bincount(t1.loc_train[:, 1], minlength=len(R.vocab))
    n_all = (
        located_in_edges(R.vocab)["code"]
        .map(R.vocab.code2idx)
        .value_counts()
        .reindex(range(len(R.vocab)), fill_value=0)
        .to_numpy()
    )
    kge = json.loads((root / "kge_reproduction.json").read_text())
    sanity = json.loads((root / "metrics" / "sanity.json").read_text())
    ga = json.loads((split_root(P1_DIR) / "split_manifest.json").read_text())["gate_a"]
    out = [
        "# CellKG Track 2 — comparison tables",
        "",
        f"Protocol label for P1: `{R.protocol}` (Gate A {'passed' if ga['passed'] else 'failed'}); "
        f"Gate B tier: `{kge['tier']}`. Values are mean ± sd over model seeds where more than one seed ran; "
        "95 % intervals are gene-level bootstrap percentiles (1,000 draws) of the seed-20260702 run. "
        "C14 = the 14 organelles with support (primary); C22 = all 22 (the published candidate set).",
        "",
    ]
    t1df = table_t1(R)
    out += ["## T1 — continuity on the 200 published held-out edges (P1)", "", md_table(t1df), ""]
    for reg in ("strict", "full"):
        d = table_metrics(R, "p1_full", reg)
        out += [
            f"## T2 — P1 full test set ({reg} regime; 1,512 held-out edges)",
            "",
            md_table(d),
            "",
        ]
        d.to_csv(root / f"comparison_p1_{reg}.csv", index=False)
    t2 = pd.concat([table_metrics(R, "p1_full", r).assign(Regime=r) for r in ("strict", "full")])
    t2.to_csv(root / "comparison_p1.csv", index=False)
    for es, reg, lab in (("p1_full", "full", "P1-full"), ("p2_pooled", "pooled", "pooled P2")):
        n_tr = n_train_p1 if es == "p1_full" else n_all
        d = table_t3(R, es, reg, n_tr)
        out += [
            f"## T3 — per organelle ({lab}; n_train = {'P1 training edges' if es == 'p1_full' else 'all edges'})",
            "",
            md_table(d),
            "",
        ]
    t4 = table_metrics(R, "p2_pooled", "pooled")
    t4.to_csv(root / "comparison_p2.csv", index=False)
    out += ["## T4 — P2 pooled out-of-fold (13,137 localized genes; 5 folds)", "", md_table(t4), ""]
    poor = []
    for m in ORDER:
        p = R.point(m, "pooled", "p2_pooled")
        if p and "info_poor" in p:
            ip = p["info_poor"]
            poor.append(
                {
                    "Method": NAMES[m],
                    "genes": ip["n_genes"],
                    "micro-MRR": fmt(ip["MRR_a0@C14"]),
                    "macro-MRR": fmt(ip["MRR_a1@C14"]),
                    "Hits@1": fmt(ip["Hits@1_a0@C14"]),
                    "mAP": fmt(ip["mAP"]),
                }
            )
    if poor:
        out += [
            "### T4b — information-poor stratum (test genes with no interaction and no keyword edge)",
            "",
            md_table(pd.DataFrame(poor)),
            "",
        ]
    fold_sd = []
    for m in ORDER:
        p = R.point(m, "pooled", "p2_pooled")
        if p and "fold_sd" in p:
            fold_sd.append(
                {
                    "Method": NAMES[m],
                    **{
                        lab: f"{p['fold_mean'][k]:.3f} ± {p['fold_sd'][k]:.3f}"
                        for k, lab in (
                            ("MRR_a0@C14", "micro-MRR"),
                            ("MRR_a1@C14", "macro-MRR"),
                            ("mAP", "mAP"),
                        )
                        if k in p["fold_sd"]
                    },
                }
            )
    if fold_sd:
        out += [
            "### T4c — per-fold mean ± sd (seed 20260702)",
            "",
            md_table(pd.DataFrame(fold_sd)),
            "",
        ]
    for es, reg, lab in (("p1_full", "full", "P1-full"), ("p2_pooled", "pooled", "pooled P2")):
        f = factorial(R, es, reg)
        if f is not None:
            out += [f"## T5 — factorial 2 × 2 ({lab}; paired gene bootstrap)", "", md_table(f), ""]
    t5b = []
    for es, reg, lab in (("p1_full", "full", "P1-full"), ("p2_pooled", "pooled", "pooled P2")):
        f = factorial(R, es, reg, FACTORIAL_XPERM, "attributes")
        if f is not None:
            out += [
                f"## T5b — factorial 2 × 2 encoder × attributes ({lab}; real vs row-permuted X; "
                "paired gene bootstrap)",
                "",
                md_table(f),
                "",
            ]
            t5b.append(f.assign(Eval=f"{es}/{reg}"))
    if t5b:
        pd.concat(t5b).to_csv(root / "comparison_factorial_xperm.csv", index=False)
    pairs = []
    for es, reg in (("p1_full", "full"), ("p1_full", "strict"), ("p2_pooled", "pooled")):
        for a, b in (
            ("sage_la", "prior"),
            ("sage_la", "RotatE"),
            ("sage_bce", "prior"),
            ("sage_la", "ppi_vote"),
            ("sage_la", "sage_bce"),
        ):
            pairs += paired(R, es, reg, a, b)
    for es, reg in (("p1_full", "full"), ("p1_full", "strict"), ("p2_pooled", "pooled")):
        for a, b in XPERM_PAIRS:  # the topology-only control, appended after the 5 original pairs
            pairs += paired(R, es, reg, a, b)
    if pairs:
        pdf = pd.DataFrame(pairs)
        pdf.to_csv(root / "comparison_paired.csv", index=False)
        out += ["## Paired differences (RQ1 / RQ3)", "", md_table(pdf), ""]
    t6 = [
        {
            "Check": "Gate A (split identity)",
            "Result": "passed" if ga["passed"] else "failed",
            "Observed": f"degree MRR {ga['observed']['MRR']:.4f}, Hits@1 {ga['observed']['Hits@1']:.4f}, "
            f"Hits@3 {ga['observed']['Hits@3']:.4f}; sizes {tuple(ga['sizes'])}",
        }
    ]
    for m, v in kge["models"].items():
        t6.append(
            {
                "Check": f"Gate B — {m}",
                "Result": v["tier"],
                "Observed": f"loc MRR {v['observed_loc']['MRR']:.4f} (|Δ| {v['abs_diff_loc_MRR']:.4f}); overall MRR "
                f"{v['observed_overall']['MRR']:.4f}; overall within tol: {v['overall_within_tol']}",
            }
        )
    for k, v in sanity.items():
        detail = {kk: vv for kk, vv in v.items() if kk != "passed"}
        t6.append(
            {
                "Check": k,
                "Result": "passed" if v.get("passed") else "FAILED",
                "Observed": json.dumps(detail, default=str)[:300],
            }
        )
    out += ["## T6 — gates and sanity log", "", md_table(pd.DataFrame(t6)), ""]
    panel = []
    for key, lab in (
        (f"sage_la_{R.protocol}_full", "SAGE-LA (P1-full)"),
        (f"RotatE_{R.protocol}_strict", "RotatE (P1-strict)"),
    ):
        p = root / "predictions" / f"panel_{key}.csv"
        if p.exists():
            panel.append(pd.read_csv(p).assign(Model=lab))
    if panel:
        out += [
            "## Panel — the 8 genes of Track 1",
            "",
            md_table(pd.concat(panel)[["Model", "symbol", "known", "top3_novel", "CEL_rank_C22"]]),
            "",
        ]
    (root / "comparison_table.md").write_text("\n".join(out) + "\n")
    log("comparison_table.md written")
    fig_f1(R, fdir / "F1_micro_vs_macro.png")
    fig_f2(R, fdir / "F2_skill_per_organelle.png")
    f3 = fig_f3(R, fdir / "F3_embeddings.png")
    fig_f4(R, fdir / "F4_training_curves.png")
    fig_f5(R, fdir / "F5_pr_rare.png")
    log(f"figures written (F3 {'rendered' if f3 else 'skipped: no embedding'})")
    update_manifest(
        "compare",
        {
            "tables": ["T1", "T2", "T3", "T4", "T5", "T5b", "T6", "panel"],
            "figures": sorted(p.name for p in fdir.glob("*.png")),
        },
        cfg,
    )
    log("OK gnn.compare")


if __name__ == "__main__":
    main()
