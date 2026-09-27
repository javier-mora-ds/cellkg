"""data/import CSVs -> node index, typed edge arrays and PyTorch Geometric edge dictionaries.

The full graph is assembled once (``make gnn-graph``). Each training run derives its
message-passing graph from a regime edge set (P1-strict / P1-full / P2 fold) plus the training
``LOCATED_IN`` edges it is allowed to see; the leakage assertion runs on that exact tensor.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from gnn import IMPORT
from gnn.common import (
    OrganelleVocab,
    StageLog,
    base_parser,
    out_dir,
    resolve_args,
    update_manifest,
    write_json,
)

if TYPE_CHECKING:
    import torch

# relation -> (csv file, source type, target type, Neo4j/PyKEEN relation name, label prefixes)
RELATIONS: dict[str, tuple[str, str, str, str, str, str]] = {
    "interacts": ("interacts_with.csv", "gene", "gene", "INTERACTS_WITH", "G:", "G:"),
    "participates_in": ("participates_in.csv", "gene", "bp", "PARTICIPATES_IN", "G:", "BP:"),
    "has_function": ("has_function.csv", "gene", "mf", "HAS_FUNCTION", "G:", "MF:"),
    "contains": ("contains.csv", "chromosome", "gene", "CONTAINS", "CHR:", "G:"),
    "subclass_of": ("subclass_of.csv", "cc", "cc", "SUBCLASS_OF", "C:", "C:"),
    "part_of": ("part_of.csv", "cc", "cc", "PART_OF", "C:", "C:"),
    "adjacent_to": ("adjacent_to.csv", "cc", "cc", "ADJACENT_TO", "C:", "C:"),
}
REL_BY_NAME = {v[3]: k for k, v in RELATIONS.items()}
LOC = ("gene", "located_in", "cc")
REV_LOC = ("cc", "rev_located_in", "gene")


@dataclass
class NodeIndex:
    genes: list[str]
    cc: list[str]  # GO ids, cc_nodes.csv order
    bp: list[str]
    mf: list[str]
    chromosome: list[str]

    def __post_init__(self) -> None:
        self.maps = {
            nt: {k: i for i, k in enumerate(getattr(self, nt))}
            for nt in ("genes", "cc", "bp", "mf", "chromosome")
        }

    @property
    def gene(self) -> dict[str, int]:
        return self.maps["genes"]

    def of(self, ntype: str) -> dict[str, int]:
        return self.maps["genes" if ntype == "gene" else ntype]

    def num_nodes(self) -> dict[str, int]:
        return {
            "gene": len(self.genes),
            "cc": len(self.cc),
            "bp": len(self.bp),
            "mf": len(self.mf),
            "chromosome": len(self.chromosome),
        }

    @classmethod
    def build(cls) -> "NodeIndex":
        rd = lambda f: pd.read_csv(IMPORT / f, dtype=str).iloc[:, 0].tolist()  # noqa: E731
        return cls(
            rd("gene_nodes.csv"),
            rd("cc_nodes.csv"),
            rd("bp_nodes.csv"),
            rd("mf_nodes.csv"),
            rd("chromosome_nodes.csv"),
        )

    def save(self, path: Path) -> None:
        write_json(path, {k: getattr(self, k) for k in ("genes", "cc", "bp", "mf", "chromosome")})

    @classmethod
    def load(cls, path: Path | None = None) -> "NodeIndex":
        path = path or out_dir() / "graph" / "node_index.json"
        return cls(**json.loads(Path(path).read_text())) if path.exists() else cls.build()

    def organelle_cc_idx(self, vocab: OrganelleVocab) -> np.ndarray:
        """cc-node index of each of the 22 organelles (score-matrix column -> cc node)."""
        return np.array([self.cc.index(g) for g in vocab.go_ids], dtype=np.int64)


def csv_edges(index: NodeIndex) -> dict[str, np.ndarray]:
    """Every non-LOCATED_IN relation (ENCODES excluded) as int64 [2, E]; PPI scores under 'interacts_score'."""
    out: dict[str, np.ndarray] = {}
    for rel, (fname, st, dt, *_rest) in RELATIONS.items():
        d = pd.read_csv(IMPORT / fname, dtype=str)
        src = d.iloc[:, 0].map(index.of(st))
        dst = d.iloc[:, 1].map(index.of(dt))
        assert src.notna().all() and dst.notna().all(), f"{fname}: dangling endpoint"
        out[rel] = np.vstack([src.to_numpy(np.int64), dst.to_numpy(np.int64)])
        if rel == "interacts":
            out["interacts_score"] = d.iloc[:, 3].astype(float).to_numpy()
    return out


def edges_from_triples(
    triples: np.ndarray, index: NodeIndex, ppi_score: dict | None = None
) -> dict[str, np.ndarray]:
    """Labelled PyKEEN triples (h, r, t) -> regime edge arrays (LOCATED_IN and ENCODES skipped)."""
    out: dict[str, np.ndarray] = {}
    for rel, (_f, st, dt, name, hp, tp) in RELATIONS.items():
        sel = triples[triples[:, 1] == name]
        src = np.array([index.of(st)[h[len(hp) :]] for h in sel[:, 0]], dtype=np.int64)
        dst = np.array([index.of(dt)[t[len(tp) :]] for t in sel[:, 2]], dtype=np.int64)
        out[rel] = np.vstack([src, dst]) if len(sel) else np.zeros((2, 0), dtype=np.int64)
        if rel == "interacts":
            out["interacts_score"] = (
                np.array([ppi_score[(a, b)] for a, b in zip(src, dst)])
                if ppi_score is not None
                else np.full(len(src), 850.0)
            )
    return out


def ppi_score_map(full: dict[str, np.ndarray]) -> dict[tuple[int, int], float]:
    ei, sc = full["interacts"], full["interacts_score"]
    return {(int(a), int(b)): float(s) for a, b, s in zip(ei[0], ei[1], sc)}


def edge_index_dict(edges: dict[str, np.ndarray]) -> dict[tuple[str, str, str], "torch.Tensor"]:
    """Static (non-localization) message-passing edge types with explicit reverse types."""
    import torch

    t = lambda a: torch.as_tensor(a, dtype=torch.long)  # noqa: E731
    ppi = edges["interacts"]
    adj = edges["adjacent_to"]
    eid = {
        ("gene", "interacts", "gene"): t(np.hstack([ppi, ppi[::-1]])),  # symmetrized once
        ("cc", "adjacent_to", "cc"): t(np.hstack([adj, adj[::-1]])),
    }
    for rel in ("participates_in", "has_function", "contains", "subclass_of", "part_of"):
        _f, st, dt, *_ = RELATIONS[rel]
        eid[(st, rel, dt)] = t(edges[rel])
        eid[(dt, f"rev_{rel}", st)] = t(edges[rel][::-1].copy())
    return eid


def loc_edge_index(
    loc: np.ndarray, org_cc: np.ndarray
) -> dict[tuple[str, str, str], "torch.Tensor"]:
    """Training localization edges (gene_idx, organelle column) -> located_in + reverse tensors."""
    import torch

    ei = (
        np.vstack([loc[:, 0], org_cc[loc[:, 1]]]).astype(np.int64)
        if len(loc)
        else np.zeros((2, 0), np.int64)
    )
    return {LOC: torch.as_tensor(ei), REV_LOC: torch.as_tensor(ei[::-1].copy())}


def ppi_edge_weight(edges: dict[str, np.ndarray], n_genes: int) -> np.ndarray:
    """(score - 700)/299 on the symmetrized PPI edges, normalized per destination (weighted mean)."""
    w = np.tile((edges["interacts_score"] - 700.0) / 299.0, 2)
    dst = np.hstack([edges["interacts"][1], edges["interacts"][0]])
    tot = np.bincount(dst, weights=w, minlength=n_genes)
    return np.where(tot[dst] > 0, w / np.maximum(tot[dst], 1e-12), 0.0)


def assert_no_leakage(loc_ei: "torch.Tensor", held: np.ndarray, org_cc: np.ndarray) -> int:
    """Raise if any held-out (gene_idx, organelle column) pair is a located_in edge of the input graph."""
    if len(held) == 0:
        return 0
    present = set(zip(loc_ei[0].tolist(), loc_ei[1].tolist()))
    heldset = {(int(g), int(org_cc[o])) for g, o in held}
    n = len(heldset & present)
    assert n == 0, f"{n} held-out localization edges present in the input graph"
    return 0


def main() -> None:
    ap = base_parser(
        "Assemble the heterogeneous graph and the gene feature table from data/import/."
    )
    ap.add_argument("--parity", action="store_true", help="compare with the persisted Neo4j export")
    args = ap.parse_args()
    cfg = resolve_args(args)
    import torch
    from gnn.features import static_features

    log = StageLog("graph")
    vocab = OrganelleVocab.load()
    index = NodeIndex.build()
    gdir = out_dir() / "graph"
    gdir.mkdir(parents=True, exist_ok=True)
    index.save(gdir / "node_index.json")
    full = csv_edges(index)
    counts = index.num_nodes()
    log(f"nodes: {counts}")
    assert counts == {"gene": 19_702, "cc": 52, "bp": 232, "mf": 129, "chromosome": 25}, counts
    eid = edge_index_dict(full)
    n_ppi = eid[("gene", "interacts", "gene")].shape[1]
    w = ppi_edge_weight(full, counts["gene"])
    raw_w = (full["interacts_score"] - 700) / 299
    log(
        f"interaction layer: {full['interacts'].shape[1]} pairs -> {n_ppi} directed edges; "
        f"raw weights in [{raw_w.min():.3f}, {raw_w.max():.3f}]"
    )
    assert n_ppi == 471_366 and raw_w.min() >= 0 and raw_w.max() <= 1
    pairs = set(map(tuple, full["interacts"].T.tolist()))
    assert not any((b, a) in pairs for a, b in pairs), "reciprocal PPI duplicates"
    assert not any(a == b for a, b in pairs), "PPI self-loops"
    from gnn.common import located_in_edges

    loc = located_in_edges(vocab)
    loc_arr = np.vstack(
        [loc["gene"].map(index.gene).to_numpy(), loc["code"].map(vocab.code2idx).to_numpy()]
    ).T
    org_cc = index.organelle_cc_idx(vocab)
    torch.save(
        {
            "edge_index": {"|".join(k): v for k, v in eid.items()},
            "loc_all": loc_edge_index(loc_arr, org_cc)[LOC],
            "ppi_weight": torch.tensor(w),
            "num_nodes": counts,
        },
        gdir / "hetero_full.pt",
    )
    feats, spec = static_features(index)
    feats.to_parquet(gdir / "features_gene.parquet")
    write_json(gdir / "feature_spec.json", spec)
    log(
        f"feature blocks: {spec['blocks']} (static width {spec['static_width']}, "
        f"+{spec['blocks']['degrees']} regime-dependent degree columns)"
    )
    edge_counts = {"|".join(k): int(v.shape[1]) for k, v in eid.items()}
    edge_counts["gene|located_in|cc (all)"] = int(len(loc_arr))
    log(f"edge types: {edge_counts}")
    rec = {"nodes": counts, "edges": edge_counts, "feature_spec": spec["blocks"]}
    if args.parity:
        from gnn.pykeen_replay import export_parity

        rec["parity"] = export_parity(log)
    update_manifest("graph", rec, cfg)
    log("OK gnn.graph")


if __name__ == "__main__":
    main()
