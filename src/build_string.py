#!/usr/bin/env python3
"""Phase 2, Task 2.4 — stream STRING and map ENSP -> ENSG.

Builds high-confidence gene-gene INTERACTS_WITH edges (combined_score >= threshold),
resolving STRING protein identifiers to Ensembl ENSG via the aliases file, dropping
unmapped endpoints, self-loops, and genes absent from the Ensembl gene table, and
canonicalizing each undirected pair once (keeping the maximum score).
"""
import gzip
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
THRESHOLD = int(os.environ.get("STRING_THRESHOLD", "700"))

LINKS = RAW / "9606.protein.links.v12.0.txt.gz"
ALIASES = RAW / "9606.protein.aliases.v12.0.txt.gz"


def build_ensp2ensg(valid_ensg):
    """string_protein_id -> ENSG, restricted to genes in the Ensembl table."""
    ensp2ensg = {}
    with gzip.open(ALIASES, "rt") as fh:
        next(fh, None)  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sp, alias, _source = parts[0], parts[1], parts[2]
            if alias.startswith("ENSG") and alias in valid_ensg and sp not in ensp2ensg:
                ensp2ensg[sp] = alias
    return ensp2ensg


def main():
    valid_ensg = set(
        pd.read_parquet(INTERIM / "genes.parquet", columns=["ensembl_id"])["ensembl_id"]
    )
    ensp2ensg = build_ensp2ensg(valid_ensg)
    print(f"ENSP->ENSG map: {len(ensp2ensg)} proteins mapped to genes in the table")

    best = {}  # (a,b) sorted -> max score
    total = kept = 0
    with gzip.open(LINKS, "rt") as fh:
        next(fh, None)  # header: protein1 protein2 combined_score
        for line in fh:
            total += 1
            p1, p2, score = line.split()
            s = int(score)
            if s < THRESHOLD:
                continue
            a = ensp2ensg.get(p1)
            b = ensp2ensg.get(p2)
            if a is None or b is None or a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if s > best.get(key, 0):
                best[key] = s
            kept += 1

    rows = [{"source_ensg": a, "target_ensg": b, "score": s} for (a, b), s in best.items()]
    df = pd.DataFrame(rows)
    df.to_parquet(INTERIM / "string_interacts.parquet", index=False)

    print(f"links scanned: {total}")
    print(f"rows >= {THRESHOLD} after mapping: {kept}")
    print(f"unique undirected INTERACTS_WITH edges: {len(df)}")
    if len(df):
        print(f"score min/max: {df['score'].min()} / {df['score'].max()}")
    assert len(df) == 0 or df["score"].min() >= THRESHOLD
    assert set(df["source_ensg"]) <= valid_ensg and set(df["target_ensg"]) <= valid_ensg
    print("OK build_string")


if __name__ == "__main__":
    main()
