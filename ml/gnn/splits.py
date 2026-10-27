"""Protocols and splits: P1 = exact replay of the PyKEEN partition (Gate A); P2 = 5 stratified gene folds.

Also the `Task` view consumed by the baselines, GraphSAGE and the KGE re-training: the regime's
message-passing edges plus the training / validation / test localization edges as
(gene index, organelle column) pairs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from gnn.common import (
    OrganelleVocab,
    StageLog,
    base_parser,
    located_in_edges,
    out_dir,
    resolve_args,
    update_manifest,
    write_json,
)
from gnn.graph import NodeIndex, csv_edges, edges_from_triples, ppi_score_map
from gnn.pykeen_replay import ensure_export, gate_a, replay, triples_fingerprint

P1_DIR = "p1_seed20260702"
P2_DIR = "p2_seed20260702"


def split_root(name: str) -> Path:
    return out_dir() / "splits" / name


# ---------------------------------------------------------------- persistence helpers
def _loc_frame(triples: np.ndarray, go2code: dict[str, str]) -> pd.DataFrame:
    sel = triples[triples[:, 1] == "LOCATED_IN"]
    return pd.DataFrame(
        {"gene": [h[2:] for h in sel[:, 0]], "code": [go2code[t[2:]] for t in sel[:, 2]]}
    )


def _save_triples(path: Path, triples: np.ndarray) -> None:
    pd.DataFrame(triples, columns=["h", "r", "t"]).to_csv(
        path, sep="\t", index=False, compression="gzip"
    )


def load_triples(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep="\t", dtype=str).to_numpy(dtype=str)


def kge_pool(export: pd.DataFrame) -> pd.DataFrame:
    """All triples of the export except ENCODES (the P1-full / P2 universe)."""
    from run_pykeen import EXCLUDE_REL

    return export[~export["r"].isin(EXCLUDE_REL)].reset_index(drop=True)


def _drop_loc(pool: pd.DataFrame, held: pd.DataFrame) -> np.ndarray:
    key = set(zip("G:" + held["gene"], "C:" + held["go_id"]))
    is_loc = pool["r"] == "LOCATED_IN"
    drop = is_loc & pd.Series([(h, t) in key for h, t in zip(pool["h"], pool["t"])])
    return pool[~drop][["h", "r", "t"]].to_numpy(dtype=str)


# ---------------------------------------------------------------- P1
def build_p1(cfg: dict, log: StageLog, vocab: OrganelleVocab) -> dict:
    ex = cfg["export"]
    df, src = ensure_export(ex["source"], log)
    res = replay(
        df,
        seed=cfg["seed"],
        interact_sample=ex["interact_sample"],
        eval_triples=ex["eval_triples"],
        loc_sample=ex["loc_sample"],
        source=src,
    )
    ga = gate_a(res, vocab.go_ids)
    log(
        f"Gate A: passed={ga['passed']} protocol={ga['protocol']} observed={ga['observed']} sizes={ga['sizes']}"
    )
    d = split_root(P1_DIR)
    (d / "tf_binary").mkdir(parents=True, exist_ok=True)
    for name, f in (
        ("train", res.train),
        ("valid", res.valid),
        ("test", res.test),
        ("test_eval", res.test_eval),
    ):
        f.to_path_binary(d / "tf_binary" / name)
    for name, f in (("train", res.train), ("valid", res.valid), ("test", res.test)):
        _save_triples(d / f"{name}_triples.tsv.gz", f.triples)
    _save_triples(d / "test_eval_3000.tsv.gz", res.test_eval.triples)
    go2code = vocab.go2code
    s200 = pd.DataFrame(
        {
            "gene": [h[2:] for h in res.sample200[:, 0]],
            "code": [go2code[t[2:]] for t in res.sample200[:, 2]],
        }
    )
    s200.to_csv(d / "eval_edges_200.csv", index=False)
    locs = {
        n: _loc_frame(f.triples, go2code)
        for n, f in (("train", res.train), ("valid", res.valid), ("test", res.test))
    }
    for n, f in locs.items():
        f.to_csv(d / f"{n}_located_in.csv", index=False)
    # P1-full KGE training triples: every non-ENCODES triple minus the held-out localization edges
    held = pd.concat([locs["valid"], locs["test"]])
    held["go_id"] = held["code"].map(dict(zip(vocab.codes, vocab.go_ids)))
    full = _drop_loc(kge_pool(df), held)
    _save_triples(d / "kge_full_train_triples.tsv.gz", full)
    man = {
        "gate_a": ga,
        "export_source": src,
        "export_set_sha256": triples_fingerprint(df),
        "n_located_in": {n: len(f) for n, f in locs.items()},
        "n_sample200": len(s200),
        "n_kge_full_train": int(len(full)),
        "fingerprints": {
            n: triples_fingerprint(pd.DataFrame(f.triples, columns=["h", "r", "t"]))
            for n, f in (("train", res.train), ("valid", res.valid), ("test", res.test))
        },
        "sample200_sha256": triples_fingerprint(
            pd.DataFrame(res.sample200, columns=["h", "r", "t"])
        ),
    }
    write_json(d / "split_manifest.json", man)
    log(
        f"P1: LOCATED_IN train/valid/test = {man['n_located_in']}; sample200 = {len(s200)}; "
        f"P1-full KGE training triples = {len(full)}"
    )
    return man


# ---------------------------------------------------------------- P2
def build_p2(cfg: dict, log: StageLog, vocab: OrganelleVocab) -> dict:
    try:
        from iterstrat.ml_stratifiers import (
            MultilabelStratifiedKFold,
            MultilabelStratifiedShuffleSplit,
        )

        stratifier = "iterative-stratification"
    except ImportError as exc:  # pragma: no cover — documented fallback not needed on this host
        raise SystemExit("iterative-stratification missing: run `make gnn-setup`") from exc
    seed, k_folds = cfg["seed"], cfg["p2"]["n_folds"]
    loc = located_in_edges(vocab)
    genes = sorted(loc["gene"].unique())
    supported = [c for c in vocab.codes if c in set(loc["code"])]
    Y = (
        pd.crosstab(loc["gene"], loc["code"])
        .reindex(index=genes, columns=supported, fill_value=0)
        .to_numpy()
        > 0
    )
    export, _ = ensure_export(cfg["export"]["source"], log)
    pool = kge_pool(export)
    code2go = dict(zip(vocab.codes, vocab.go_ids))
    root = split_root(P2_DIR)
    mskf = MultilabelStratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    seen: list[str] = []
    fold_stats = []
    for k, (rest_idx, test_idx) in enumerate(mskf.split(np.zeros(len(genes)), Y), start=1):
        sss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed + k)
        tr_rel, va_rel = next(sss.split(np.zeros(len(rest_idx)), Y[rest_idx]))
        part = pd.DataFrame({"gene": genes})
        part["partition"] = ""
        part.loc[rest_idx[tr_rel], "partition"] = "train"
        part.loc[rest_idx[va_rel], "partition"] = "valid"
        part.loc[test_idx, "partition"] = "test"
        fd = root / f"fold_{k}"
        fd.mkdir(parents=True, exist_ok=True)
        part.to_csv(fd / "gene_partition.csv", index=False)
        pmap = dict(zip(part["gene"], part["partition"]))
        for p in ("train", "valid", "test"):
            loc[loc["gene"].map(pmap) == p][["gene", "code"]].to_csv(
                fd / f"{p}_edges.csv", index=False
            )
        held = loc[loc["gene"].map(pmap).isin(["valid", "test"])].copy()
        held["go_id"] = held["code"].map(code2go)
        kt = _drop_loc(pool, held)
        _save_triples(fd / "kge_train_triples.tsv.gz", kt)
        test_pos = Y[test_idx].sum(0)
        seen += [genes[i] for i in test_idx]
        fold_stats.append(
            {
                "fold": k,
                "n_train": int(len(tr_rel)),
                "n_valid": int(len(va_rel)),
                "n_test": int(len(test_idx)),
                "kge_train_triples": int(len(kt)),
                "test_positives": dict(zip(supported, test_pos.tolist())),
            }
        )
        log(
            f"P2 fold {k}: train/valid/test genes = {len(tr_rel)}/{len(va_rel)}/{len(test_idx)}; "
            f"KGE training triples = {len(kt)}; LYS/PRX/CRM test = "
            f"{dict((c, int(test_pos[supported.index(c)])) for c in ('LYS', 'PRX', 'CRM'))}"
        )
    assert (
        len(seen) == len(set(seen)) == len(genes) == 13_137
    ), "P2 folds not a partition of the localized genes"
    support = Y.sum(0)
    for fs in fold_stats:
        for c, n in zip(supported, support):
            if n >= 10:
                assert fs["test_positives"][c] > 0, f"fold {fs['fold']}: no {c} test positive"
    man = {
        "stratifier": stratifier,
        "n_genes": len(genes),
        "supported": supported,
        "support": dict(zip(supported, support.tolist())),
        "folds": fold_stats,
    }
    write_json(root / "split_manifest.json", man)
    return man


# ---------------------------------------------------------------- task view
@dataclass
class Task:
    protocol: str  # p1 | p1r | p2
    regime: str  # strict | full | fold_k
    edges: dict  # non-localization message-passing edges
    loc_train: np.ndarray  # [E, 2] (gene index, organelle column)
    loc_valid: np.ndarray
    loc_test: np.ndarray
    train_genes: np.ndarray  # loss-row candidates
    valid_genes: np.ndarray
    test_genes: np.ndarray
    kge_source: str  # "binary:<dir>" or a triples file

    @property
    def name(self) -> str:
        return f"{self.protocol}_{self.regime}"

    def labels(self, which: str, n_genes: int, n_org: int) -> np.ndarray:
        Y = np.zeros((n_genes, n_org), dtype=bool)
        e = getattr(self, f"loc_{which}")
        Y[e[:, 0], e[:, 1]] = True
        return Y


def _edge_arr(df: pd.DataFrame, index: NodeIndex, vocab: OrganelleVocab) -> np.ndarray:
    if len(df) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.vstack(
        [df["gene"].map(index.gene).to_numpy(), df["code"].map(vocab.code2idx).to_numpy()]
    ).T.astype(np.int64)


def p1_protocol() -> str:
    man = json.loads((split_root(P1_DIR) / "split_manifest.json").read_text())
    return man["gate_a"]["protocol"]


def list_tasks(cfg: dict) -> list[tuple[str, str]]:
    out = [("p1", r) for r in cfg["graph"]["regimes"]] if "p1" in cfg["protocols"] else []
    if "p2" in cfg["protocols"]:
        out += [("p2", f"fold_{k}") for k in range(1, cfg["p2"]["n_folds"] + 1)]
    return out


def make_task(
    protocol: str, regime: str, index: NodeIndex | None = None, vocab: OrganelleVocab | None = None
) -> Task:
    index = index or NodeIndex.load()
    vocab = vocab or OrganelleVocab.load()
    full = csv_edges(index)
    if protocol.startswith("p1"):
        d = split_root(P1_DIR)
        locs = {
            p: _edge_arr(pd.read_csv(d / f"{p}_located_in.csv"), index, vocab)
            for p in ("train", "valid", "test")
        }
        if regime == "strict":
            edges = edges_from_triples(
                load_triples(d / "train_triples.tsv.gz"), index, ppi_score_map(full)
            )
            kge = f"binary:{d / 'tf_binary'}"
        else:
            edges = full
            kge = str(d / "kge_full_train_triples.tsv.gz")
        return Task(
            p1_protocol(),
            regime,
            edges,
            locs["train"],
            locs["valid"],
            locs["test"],
            np.unique(locs["train"][:, 0]),
            np.unique(locs["valid"][:, 0]),
            np.unique(locs["test"][:, 0]),
            kge,
        )
    fd = split_root(P2_DIR) / regime
    part = pd.read_csv(fd / "gene_partition.csv")
    locs = {
        p: _edge_arr(pd.read_csv(fd / f"{p}_edges.csv"), index, vocab)
        for p in ("train", "valid", "test")
    }
    gid = lambda p: np.sort(
        part.loc[part["partition"] == p, "gene"].map(index.gene).to_numpy()
    )  # noqa: E731
    return Task(
        "p2",
        regime,
        full,
        locs["train"],
        locs["valid"],
        locs["test"],
        gid("train"),
        gid("valid"),
        gid("test"),
        str(fd / "kge_train_triples.tsv.gz"),
    )


def main() -> None:
    ap = base_parser("P1 exact PyKEEN replay (Gate A) and P2 stratified gene folds.")
    args = ap.parse_args()
    cfg = resolve_args(args)
    log = StageLog("splits")
    vocab = OrganelleVocab.load()
    rec = {}
    if "p1" in cfg["protocols"]:
        rec["p1"] = build_p1(cfg, log, vocab)
    if "p2" in cfg["protocols"]:
        rec["p2"] = {k: v for k, v in build_p2(cfg, log, vocab).items() if k != "folds"}
    update_manifest("splits", rec, cfg)
    log("OK gnn.splits")


if __name__ == "__main__":
    main()
