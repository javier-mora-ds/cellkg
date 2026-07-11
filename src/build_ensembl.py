#!/usr/bin/env python3
"""Phase 2, Task 2.1 — parse Ensembl gene lines only into data/interim/genes.parquet.

Extracts one row per `feature == "gene"` from the GRCh38 GTF, reading gene_biotype
from the 9th attribute column by regex (never the full-feature dump). Parsing is
null-tolerant: ~45.7% of genes carry no gene_name, so `symbol` may be None and every
gene is keyed strictly on its Ensembl ENSG.
"""
import gzip
import re
from glob import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

CANON_CHROM = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}
ID_RE = re.compile(r'gene_id "([^"]+)"')
NAME_RE = re.compile(r'gene_name "([^"]+)"')
BIO_RE = re.compile(r'gene_biotype "([^"]+)"')


def gtf_path() -> Path:
    hits = sorted(glob(str(RAW / "Homo_sapiens.GRCh38.*.gtf.gz")))
    if not hits:
        raise FileNotFoundError("no Homo_sapiens.GRCh38.*.gtf.gz in data/raw/")
    return Path(hits[0])


def main():
    gtf = gtf_path()
    rows = []
    with gzip.open(gtf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9 or c[2] != "gene":
                continue
            attrs = c[8]
            gid = ID_RE.search(attrs)
            if gid is None:
                continue
            name = NAME_RE.search(attrs)
            bio = BIO_RE.search(attrs)
            start, end = int(c[3]), int(c[4])
            rows.append(
                {
                    "ensembl_id": gid.group(1).split(".")[0],
                    "symbol": name.group(1) if name else None,
                    "biotype": bio.group(1) if bio else None,
                    "chromosome": c[0],
                    "start": start,
                    "end": end,
                    "strand": c[6],
                    "length": end - start + 1,
                }
            )

    genes = pd.DataFrame(rows).drop_duplicates("ensembl_id").reset_index(drop=True)

    groups = pd.read_csv(RAW / "organelles" / "biotype_groups.csv")
    genes = genes.merge(groups, on="biotype", how="left")
    genes["biotype_group"] = genes["biotype_group"].fillna("OtherRNA")

    genes.to_parquet(INTERIM / "genes.parquet", index=False)

    n = len(genes)
    null_sym = genes["symbol"].isna().mean()
    chroms = sorted(genes["chromosome"].unique())
    canon = [c for c in chroms if c in CANON_CHROM]
    print(f"source: {gtf.name}")
    print(f"genes: {n}")
    print(f"symbol null rate: {null_sym:.3f}")
    print(f"canonical chromosomes present: {len(canon)} -> {canon}")
    print("biotype_group counts:")
    print(genes["biotype_group"].value_counts().to_string())
    assert n == 78724, f"expected 78724 genes, got {n}"
    assert set(canon) == CANON_CHROM, f"missing canonical chromosomes: {CANON_CHROM - set(canon)}"
    print("OK build_ensembl")


if __name__ == "__main__":
    main()
