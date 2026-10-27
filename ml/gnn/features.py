"""Gene attribute vector (plan §5.C): static blocks from the CSVs plus regime-dependent degrees."""
from __future__ import annotations

import numpy as np
import pandas as pd
from gnn import IMPORT, RAW
from gnn.common import gene_table

TISSUE = [
    "Low tissue specificity",
    "Tissue enhanced",
    "Tissue enriched",
    "Group enriched",
    "Not detected",
]
CONTINUOUS = ["log10_length", "log1p_n_interactions"]
DEGREES = ["deg_ppi", "deg_ppi_weighted", "n_bp", "n_mf"]


def static_features(index) -> tuple[pd.DataFrame, dict]:
    """Static per-gene attributes in node-index order; continuous columns are left unstandardized."""
    g = gene_table().set_index("gene").loc[index.genes]
    groups = pd.read_csv(RAW / "organelles" / "biotype_groups.csv")
    group_vocab = sorted(groups["biotype_group"].unique())
    bt_vocab = sorted(g["biotype"].unique())
    blocks: dict[str, pd.DataFrame] = {}
    blocks["biotype"] = pd.get_dummies(pd.Categorical(g["biotype"], bt_vocab), prefix="bt")
    grp = g["biotype"].map(dict(zip(groups["biotype"], groups["biotype_group"])))
    blocks["biotype_group"] = pd.get_dummies(pd.Categorical(grp, group_vocab), prefix="grp")
    blocks["chromosome"] = pd.get_dummies(
        pd.Categorical(g["chromosome"], index.chromosome), prefix="chr"
    )
    blocks["strand"] = pd.DataFrame({"strand": np.where(g["strand"] == "+", 1.0, -1.0)})
    blocks["length"] = pd.DataFrame(
        {"log10_length": np.log10(g["length"].astype(float).clip(lower=1))}
    )
    ts = pd.get_dummies(pd.Categorical(g["rna_tissue_specificity"], TISSUE), prefix="ts")
    ts["ts_missing"] = g["rna_tissue_specificity"].isna().to_numpy()
    blocks["tissue_specificity"] = ts
    ni = g["n_interactions"]
    blocks["n_interactions"] = pd.DataFrame(
        {
            "log1p_n_interactions": np.log1p(ni.fillna(0).astype(float)),
            "n_interactions_missing": ni.isna().to_numpy(),
        }
    )
    gi = index.gene
    for rel, fname, vocab, pfx in (
        ("bp", "participates_in.csv", index.bp, "bp"),
        ("mf", "has_function.csv", index.mf, "mf"),
    ):
        d = pd.read_csv(IMPORT / fname, dtype=str)
        m = np.zeros((len(index.genes), len(vocab)), dtype=np.float32)
        col = {t: i for i, t in enumerate(vocab)}
        m[d.iloc[:, 0].map(gi).to_numpy(), d.iloc[:, 1].map(col).to_numpy()] = 1.0
        blocks[rel] = pd.DataFrame(m, columns=[f"{pfx}:{t}" for t in vocab])
    for b in blocks.values():
        b.index = range(len(index.genes))
    df = pd.concat(blocks.values(), axis=1).astype(np.float32)
    df.columns = [str(c) for c in df.columns]
    widths = {k: int(v.shape[1]) for k, v in blocks.items()}
    widths["degrees"] = len(DEGREES)
    spec = {
        "blocks": widths,
        "static_width": int(df.shape[1]),
        "total_width": int(df.shape[1]) + len(DEGREES),
        "continuous": CONTINUOUS + DEGREES,
        "biotypes": bt_vocab,
        "biotype_groups": group_vocab,
        "genes_missing_tissue": int(g["rna_tissue_specificity"].isna().sum()),
        "genes_missing_n_interactions": int(ni.isna().sum()),
    }
    return df, spec


def degree_block(edges: dict[str, np.ndarray], n_genes: int) -> np.ndarray:
    """log1p degrees on the current message-passing graph (never from localization edges)."""
    ppi = edges["interacts"]
    ends = np.hstack([ppi[0], ppi[1]])
    w = np.tile(edges["interacts_score"] / 1000.0, 2)
    out = np.vstack(
        [
            np.bincount(ends, minlength=n_genes),
            np.bincount(ends, weights=w, minlength=n_genes),
            np.bincount(edges["participates_in"][0], minlength=n_genes),
            np.bincount(edges["has_function"][0], minlength=n_genes),
        ]
    ).T
    return np.log1p(out).astype(np.float32)


def build_features(
    static: pd.DataFrame, edges: dict[str, np.ndarray], train_rows: np.ndarray
) -> np.ndarray:
    """Static blocks + regime degrees; continuous columns standardized on the training rows."""
    X = np.hstack([static.to_numpy(np.float32), degree_block(edges, len(static))])
    cols = list(static.columns) + DEGREES
    for c in CONTINUOUS + DEGREES:
        j = cols.index(c)
        mu, sd = X[train_rows, j].mean(), X[train_rows, j].std()
        X[:, j] = (X[:, j] - mu) / (sd if sd > 0 else 1.0)
    return X.astype(np.float32)
