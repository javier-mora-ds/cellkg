"""The harness over every score matrix -> reports/gnn/metrics/ (plan §5.H), sanity checks, predictions."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from gnn.common import (
    OrganelleVocab,
    ScoreMatrix,
    StageLog,
    base_parser,
    gene_table,
    out_dir,
    resolve_args,
    update_manifest,
    write_json,
)
from gnn.graph import NodeIndex
from gnn.metrics import Harness, bootstrap_weights, fit_thresholds, point
from gnn.splits import P1_DIR, P2_DIR, Task, load_triples, make_task, split_root

PANEL = ["TP53", "BRCA1", "EGFR", "MYC", "VDAC1", "GAPDH", "LMNA", "CANX"]
SCALARS_BOOT = [
    "MRR_a0@C22",
    "MRR_a0@C14",
    "MRR_a0.5@C14",
    "MRR_a1@C14",
    "Hits@1_a0@C22",
    "Hits@3_a0@C22",
    "Hits@1_a0@C14",
    "Hits@3_a0@C14",
    "Hits@10_a0@C14",
    "mAP",
    "macroAUROC",
    "LRAP",
    "macroF1",
    "microF1",
    "R_macroMRR",
    "R_microMRR",
    "U_macroMRR",
    "R_mAP",
    "U_mAP",
]
PER_CLASS_BOOT = ["MRR_o", "AP_o", "AUROC_o", "support_o"]


@dataclass
class EvalSet:
    name: str
    rows: np.ndarray  # gene indices
    Y_pos: np.ndarray  # bool [n, 22] positives to evaluate
    Y_mask: np.ndarray  # bool [n, 22] known, not evaluated
    fold: np.ndarray  # [n] fold id (P2) or 0
    info_poor: np.ndarray  # [n] bool: no interaction and no keyword edge in the regime graph


def _info_poor(task: Task, n_genes: int) -> np.ndarray:
    rich = np.zeros(n_genes, dtype=bool)
    for rel in ("interacts", "participates_in", "has_function"):
        rich[task.edges[rel][0]] = True
        if rel == "interacts":
            rich[task.edges[rel][1]] = True
    return ~rich


def p1_eval_sets(task: Task, index: NodeIndex, vocab: OrganelleVocab) -> list[EvalSet]:
    n_genes, n_org = len(index.genes), len(vocab)
    Ytr, Yva, Yte = (task.labels(p, n_genes, n_org) for p in ("train", "valid", "test"))
    poor = _info_poor(task, n_genes)
    s200 = pd.read_csv(split_root(P1_DIR) / "eval_edges_200.csv")
    g200 = np.array(sorted(set(s200["gene"].map(index.gene))))
    Yp = np.zeros((len(g200), n_org), dtype=bool)
    Yp[np.searchsorted(g200, s200["gene"].map(index.gene)), s200["code"].map(vocab.code2idx)] = True
    known = (Ytr | Yva | Yte)[g200]
    rows = task.test_genes
    return [
        EvalSet("p1_sub200", g200, Yp, known & ~Yp, np.zeros(len(g200), int), poor[g200]),
        EvalSet(
            "p1_full",
            rows,
            Yte[rows],
            (Ytr | Yva)[rows] & ~Yte[rows],
            np.zeros(len(rows), int),
            poor[rows],
        ),
    ]


def p2_eval_set(tasks: list[Task], index: NodeIndex, vocab: OrganelleVocab) -> EvalSet:
    n_genes, n_org = len(index.genes), len(vocab)
    rows, Y, fold, poor = [], [], [], []
    for k, t in enumerate(tasks, start=1):
        rows.append(t.test_genes)
        Y.append(t.labels("test", n_genes, n_org)[t.test_genes])
        fold.append(np.full(len(t.test_genes), k))
        poor.append(_info_poor(t, n_genes)[t.test_genes])
    Yp = np.vstack(Y)
    return EvalSet(
        "p2_pooled",
        np.concatenate(rows),
        Yp,
        np.zeros_like(Yp),
        np.concatenate(fold),
        np.concatenate(poor),
    )


def predictions(S: np.ndarray, task: Task, n_genes: int, n_org: int, min_pos: int) -> np.ndarray:
    """Binary predictions for every gene from thresholds fitted on the validation rows only."""
    vg = task.valid_genes
    Ytr = task.labels("train", n_genes, n_org)
    Yv = task.labels("valid", n_genes, n_org)[vg]
    Mv = Ytr[vg] if task.protocol.startswith("p1") else np.zeros_like(Yv)
    supported = Ytr[task.train_genes].any(0)
    theta = fit_thresholds(S[vg], Yv, Mv, supported, min_pos)
    return S >= theta[None, :]


class Evaluator:
    def __init__(self, cfg: dict, log: StageLog):
        self.cfg, self.log = cfg, log
        self.index, self.vocab = NodeIndex.load(), OrganelleVocab.load()
        self.n_genes, self.n_org = len(self.index.genes), len(self.vocab)
        mc = cfg["metrics"]
        self.mc = mc
        self.tasks: dict[tuple[str, str], Task] = {}
        self.sets: dict[str, EvalSet] = {}
        self.W: dict[str, np.ndarray] = {}
        self.supported = np.zeros(self.n_org, dtype=bool)

    def task(self, protocol: str, regime: str) -> Task:
        key = ("p1" if protocol.startswith("p1") else "p2", regime)
        if key not in self.tasks:
            self.tasks[key] = make_task(*key, self.index, self.vocab)
        return self.tasks[key]

    def harness(self, es: EvalSet) -> Harness:
        return Harness(
            es.Y_pos,
            es.Y_mask,
            self.vocab.codes,
            self.supported,
            self.mc["rare_tail"],
            self.mc["ultra_rare"],
            self.mc["ks"],
            self.mc["alphas"],
            self.mc["ties"],
        )

    def weights(self, es: EvalSet) -> np.ndarray:
        if es.name not in self.W:
            b = self.mc["bootstrap"]
            self.W[es.name] = bootstrap_weights(
                len(es.rows), b["draws"], self.cfg["seed"] + len(es.name)
            )
        return self.W[es.name]

    def score(
        self,
        es: EvalSet,
        S_rows: np.ndarray,
        pred_rows: np.ndarray,
        tag: str,
        boot: bool,
        extra: dict,
    ) -> dict:
        h = self.harness(es)
        m = point(h.evaluate(S_rows, pred_rows))
        m.update(extra)
        if es.info_poor.any():
            sub = es.info_poor
            hp = Harness(
                es.Y_pos[sub],
                es.Y_mask[sub],
                self.vocab.codes,
                self.supported,
                self.mc["rare_tail"],
                self.mc["ultra_rare"],
                self.mc["ks"],
                self.mc["alphas"],
                self.mc["ties"],
            )
            pm = point(hp.evaluate(S_rows[sub], pred_rows[sub]))
            m["info_poor"] = {
                "n_genes": int(sub.sum()),
                **{k: pm[k] for k in ("MRR_a0@C14", "MRR_a1@C14", "mAP", "Hits@1_a0@C14")},
            }
        write_json(out_dir() / "metrics" / f"{tag}.json", m)
        if boot:
            bm = h.evaluate(S_rows, pred_rows, self.weights(es))
            np.savez_compressed(
                out_dir() / "metrics" / f"bootstrap_{tag}.npz",
                **{k: bm[k] for k in SCALARS_BOOT + PER_CLASS_BOOT if k in bm},
            )
        return m


def _load_all() -> list[ScoreMatrix]:
    return [ScoreMatrix.load(p) for p in sorted((out_dir() / "scores").glob("*.npz"))]


def sanity_checks(ev: Evaluator, sms: list[ScoreMatrix], summary: pd.DataFrame) -> dict:
    out: dict = {}
    split_man = json.loads((split_root(P1_DIR) / "split_manifest.json").read_text())
    ga = split_man["gate_a"]
    # S1 — harness == published predict_target loop (recorded by gnn.kge)
    kr = out_dir() / "kge_reproduction.json"
    if kr.exists():
        gb = json.loads(kr.read_text())
        out["S1"] = {
            m: {"passed": v["S1_passed"], "max_abs_rr_diff": v["S1_max_abs_rr_diff"]}
            for m, v in gb["models"].items()
        }
        out["S1"]["passed"] = all(v["passed"] for k, v in out["S1"].items() if k != "passed")
    # S2 — prior reproduces the degree baseline; AUROC 0.5; AP = prevalence
    pr = summary[
        (summary.model == "prior")
        & (summary.eval_set == "p1_sub200")
        & (summary.regime == "strict")
    ]
    s2 = {}
    if len(pr):
        r = pr.iloc[0]
        s2["sub200_C22"] = {
            "MRR": r["MRR_a0@C22"],
            "Hits@1": r["Hits@1_a0@C22"],
            "Hits@3": r["Hits@3_a0@C22"],
        }
        s2["matches_gate_a"] = all(
            abs(s2["sub200_C22"][k] - ga["observed"][k]) < 1e-9 for k in ("MRR", "Hits@1", "Hits@3")
        )
        m = json.loads(
            (
                out_dir()
                / "metrics"
                / f"prior_{ga['protocol']}_full_p1_full_seed{ev.cfg['seed']}.json"
            ).read_text()
        )
        es = ev.sets["p1_full"]
        auc = np.array(m["AUROC_o"], dtype=float)
        ap = np.array(m["AP_o"], dtype=float)
        ok = ~es.Y_mask
        prev = np.array(
            [es.Y_pos[ok[:, o], o].mean() if ok[:, o].any() else np.nan for o in range(ev.n_org)]
        )
        sup = ev.supported & ~np.isnan(ap)
        s2["auroc_all_half"] = bool(np.allclose(auc[sup], 0.5))
        s2["ap_equals_prevalence"] = bool(np.allclose(ap[sup], prev[sup]))
    # unfiltered per-edge view: every positive edge ranked by the prior order -> macro-MRR = H14/14
    es = ev.sets.get("p2_pooled")
    if es is not None:
        counts = es.Y_pos.sum(0).astype(float)
        order = np.argsort(-counts, kind="mergesort")
        rank = np.empty(ev.n_org)
        rank[order] = np.arange(1, ev.n_org + 1)
        _, cols = np.nonzero(es.Y_pos)
        per = pd.Series(1 / rank[cols]).groupby(cols).mean()
        s2["prior_unfiltered_macro_MRR"] = float(per.mean())
        s2["H14_over_14"] = float(sum(1 / k for k in range(1, 15)) / 14)
        s2["macro_equals_H14"] = abs(s2["prior_unfiltered_macro_MRR"] - s2["H14_over_14"]) < 1e-9
    s2["passed"] = bool(
        s2.get("matches_gate_a", False)
        and s2.get("auroc_all_half", False)
        and s2.get("ap_equals_prevalence", False)
        and s2.get("macro_equals_H14", False)
    )
    out["S2"] = s2
    # S3 — random scores -> micro-MRR ~ H_c/c
    rng = np.random.default_rng(0)
    es = ev.sets["p2_pooled"]
    h = ev.harness(es)
    Hc = lambda c: sum(1 / k for k in range(1, c + 1)) / c  # noqa: E731
    r22, r14 = [], []
    for _ in range(20):
        m = point(h.evaluate(rng.random((len(es.rows), ev.n_org))))
        r22.append(m["MRR_a0@C22"])
        r14.append(m["MRR_a0@C14"])
    # filtered: multi-label genes have fewer effective candidates, so the expectation is slightly above H_c/c
    out["S3"] = {
        "random_micro_MRR_C22": float(np.mean(r22)),
        "H22_over_22": Hc(22),
        "random_micro_MRR_C14": float(np.mean(r14)),
        "H14_over_14": Hc(14),
        "passed": bool(abs(np.mean(r22) - Hc(22)) < 0.02 and abs(np.mean(r14) - Hc(14)) < 0.02),
    }
    # S5 — leakage: training assertions logged, KGE factories hold no held-out localization triple
    trained = [s for s in sms if s.model.startswith(("sage", "mlp"))]
    s5 = {
        "trained_runs": len(trained),
        "leakage_assert_passed": all(s.meta.get("leakage_assert") == "passed" for s in trained),
    }
    kge_ok = True
    for (p, r), t in ev.tasks.items():
        if t.kge_source.startswith("binary:"):
            tr = load_triples(split_root(P1_DIR) / "train_triples.tsv.gz")
        else:
            tr = load_triples(Path(t.kge_source))
        loc = {(h, tt) for h, rr, tt in tr if rr == "LOCATED_IN"}
        held = {
            ("G:" + ev.index.genes[g], "C:" + ev.vocab.go_ids[o])
            for g, o in np.vstack([t.loc_valid, t.loc_test])
        }
        kge_ok &= not (loc & held)
    s5["kge_factories_clean"] = bool(kge_ok)
    s5[
        "class_stats_from_training_only"
    ] = True  # ClassStats is built from task.train_genes (train.py)
    s5["passed"] = s5["leakage_assert_passed"] and s5["kge_factories_clean"]
    out["S5"] = s5
    # S4 — rho = 0 (targets visible in the input graph): near-zero training loss, no validation gain
    cdir = out_dir() / "curves"
    r0, r5 = (cdir / f"sage_la_rho0_{ga['protocol']}_full_seed{ev.cfg['seed']}.csv",
              cdir / f"sage_la_{ga['protocol']}_full_seed{ev.cfg['seed']}.csv")
    if r0.exists() and r5.exists():
        c0, c5 = pd.read_csv(r0), pd.read_csv(r5)
        out["S4"] = {"rho0_min_train_loss": float(c0["loss"].min()), "rho05_min_train_loss": float(c5["loss"].min()),
                     "rho0_best_val_map": float(c0["val_map_elig"].max()),
                     "rho05_best_val_map": float(c5["val_map_elig"].max()),
                     "passed": bool(c0["loss"].min() < c5["loss"].min()
                                    and c0["val_map_elig"].max() < c5["val_map_elig"].max())}
    # S6 — partition sizes, sample200 checksum, P2 partition
    p2m = json.loads((split_root(P2_DIR) / "split_manifest.json").read_text())
    out["S6"] = {
        "sizes": ga["sizes"],
        "entities": ga["counts"]["entities"],
        "sample200_sha256": split_man["sample200_sha256"],
        "p2_genes": p2m["n_genes"],
        "passed": tuple(ga["sizes"]) == (97133, 12142, 12142)
        and ga["counts"]["entities"] == 20140
        and p2m["n_genes"] == 13137,
    }
    return out


def novel_and_panel(ev: Evaluator, sms: dict) -> None:
    pdir = out_dir() / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    g = gene_table()
    sym = dict(zip(g["gene"], g["symbol"]))
    task = ev.task("p1", "full")
    Yall = (
        task.labels("train", ev.n_genes, ev.n_org)
        | task.labels("valid", ev.n_genes, ev.n_org)
        | task.labels("test", ev.n_genes, ev.n_org)
    )
    codes = np.array(ev.vocab.codes)
    for key, sm in sms.items():
        S = sm.S
        # novel predictions for the genes without any localization edge
        unl = np.flatnonzero(~Yall.any(1))
        top = np.argsort(-S[unl], 1, kind="mergesort")[:, :3]
        pd.DataFrame(
            {
                "gene": [ev.index.genes[i] for i in unl],
                "symbol": [sym.get(ev.index.genes[i]) for i in unl],
                **{f"top{k + 1}": codes[top[:, k]] for k in range(3)},
                **{f"score{k + 1}": S[unl, top[:, k]] for k in range(3)},
            }
        ).to_csv(pdir / f"novel_{key}.csv", index=False)
        rows = []
        for s in PANEL:
            ens = [e for e, v in sym.items() if v == s]
            if not ens:
                continue
            i = ev.index.gene[ens[0]]
            order = np.argsort(-S[i], kind="mergesort")
            known = [str(c) for c in codes[Yall[i]]]
            novel = ", ".join([f"{codes[o]} ({S[i, o]:.2f})" for o in order if not Yall[i, o]][:3])
            cel_rank = int(np.flatnonzero(codes[order] == "CEL")[0]) + 1
            rows.append(
                {
                    "symbol": s,
                    "known": " ".join(sorted(known)),
                    "top3_novel": novel,
                    "CEL_rank_C22": cel_rank,
                }
            )
        pd.DataFrame(rows).to_csv(pdir / f"panel_{key}.csv", index=False)


def main() -> None:
    ap = base_parser("Harness over every score matrix -> reports/gnn/metrics/.")
    args = ap.parse_args()
    cfg = resolve_args(args)
    log = StageLog("evaluate")
    ev = Evaluator(cfg, log)
    sms = _load_all()
    log(f"{len(sms)} score matrices")
    ev.supported = ev.task("p1", "strict").labels("train", ev.n_genes, ev.n_org).any(0)
    t1 = ev.task("p1", "full")
    for es in p1_eval_sets(t1, ev.index, ev.vocab):
        ev.sets[es.name] = es
    folds = sorted(
        {s.regime for s in sms if s.protocol == "p2"}, key=lambda r: int(r.split("_")[1])
    )
    if folds:
        ev.sets["p2_pooled"] = p2_eval_set([ev.task("p2", r) for r in folds], ev.index, ev.vocab)
    primary_seed = cfg["seed"]
    rows = []
    p2_groups: dict[tuple[str, int], dict[str, ScoreMatrix]] = defaultdict(dict)
    min_pos = cfg["metrics"]["min_valid_positives"]
    for sm in sms:
        if sm.protocol == "p2":
            p2_groups[(sm.model, sm.seed)][sm.regime] = sm
            continue
        task = ev.task(sm.protocol, sm.regime)
        pred = predictions(sm.S, task, ev.n_genes, ev.n_org, min_pos)
        for es_name in ("p1_sub200", "p1_full"):
            es = ev.sets[es_name]
            tag = f"{sm.model}_{sm.protocol}_{sm.regime}_{es_name}_seed{sm.seed}"
            extra = {"coverage_test": sm.meta.get("coverage_test", 1.0)}
            m = ev.score(es, sm.S[es.rows], pred[es.rows], tag, sm.seed == primary_seed, extra)
            rows.append(
                {
                    "model": sm.model,
                    "protocol": sm.protocol,
                    "regime": sm.regime,
                    "eval_set": es_name,
                    "seed": sm.seed,
                    **{k: v for k, v in m.items() if not isinstance(v, (list, dict))},
                }
            )
    for (model, seed), by_fold in sorted(p2_groups.items()):
        if set(by_fold) != set(folds):
            log(f"  skip {model} seed {seed}: folds {sorted(by_fold)} incomplete")
            continue
        es = ev.sets["p2_pooled"]
        S_rows = np.zeros((len(es.rows), ev.n_org), dtype=np.float32)
        P_rows = np.zeros((len(es.rows), ev.n_org), dtype=bool)
        fold_metrics = []
        for k, r in enumerate(folds, start=1):
            sm = by_fold[r]
            task = ev.task("p2", r)
            sel = es.fold == k
            S_rows[sel] = sm.S[es.rows[sel]]
            pred = predictions(sm.S, task, ev.n_genes, ev.n_org, min_pos)
            P_rows[sel] = pred[es.rows[sel]]
            fm = point(
                ev.harness(
                    EvalSet(
                        "f",
                        es.rows[sel],
                        es.Y_pos[sel],
                        es.Y_mask[sel],
                        es.fold[sel],
                        es.info_poor[sel],
                    )
                ).evaluate(S_rows[sel], P_rows[sel])
            )
            fold_metrics.append({k2: fm[k2] for k2 in SCALARS_BOOT if k2 in fm})
        fdf = pd.DataFrame(fold_metrics)
        extra = {
            "fold_mean": fdf.mean().to_dict(),
            "fold_sd": fdf.std(ddof=1).to_dict(),
            "coverage_test": float(
                np.mean([s.meta.get("coverage_test", 1.0) for s in by_fold.values()])
            ),
        }
        tag = f"{model}_p2_pooled_seed{seed}"
        m = ev.score(es, S_rows, P_rows, tag, seed == primary_seed, extra)
        rows.append(
            {
                "model": model,
                "protocol": "p2",
                "regime": "pooled",
                "eval_set": "p2_pooled",
                "seed": seed,
                **{k: v for k, v in m.items() if not isinstance(v, (list, dict))},
                **{f"fold_sd_{k}": v for k, v in extra["fold_sd"].items()},
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir() / "metrics" / "summary.csv", index=False)
    log(f"{len(summary)} metric bundles -> metrics/summary.csv")
    checks = sanity_checks(ev, sms, summary)
    write_json(out_dir() / "metrics" / "sanity.json", checks)
    log("sanity: " + ", ".join(f"{k}={v.get('passed')}" for k, v in checks.items()))
    by_name = {
        f"{s.model}_{s.protocol}_{s.regime}": s
        for s in sms
        if s.seed == primary_seed
        and (
            (s.model == "sage_la" and s.regime == "full")
            or (s.model == "RotatE" and s.regime == "strict")
        )
    }
    novel_and_panel(ev, by_name)
    update_manifest(
        "evaluate",
        {"bundles": len(summary), "sanity": {k: v.get("passed") for k, v in checks.items()}},
        cfg,
    )
    log("OK gnn.evaluate")


if __name__ == "__main__":
    main()
