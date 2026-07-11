#!/usr/bin/env python3
"""Phase 2, Task 2.2 — parse the Human Protein Atlas JSON into data/interim/.

HPA multi-value fields are native JSON arrays (no delimiter splitting). Every entry
keys on a single, unique Ensembl ENSG. Produces the LOCATED_IN (with reliability),
PARTICIPATES_IN, HAS_FUNCTION and ENCODES edge tables, the BiologicalProcess /
MolecularFunction / Protein node tables, and per-gene HPA properties
(rna_tissue_specificity, n_interactions).
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)


def as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def main():
    with (RAW / "proteinatlas.json").open() as fh:
        data = json.load(fh)

    xwalk = dict(
        pd.read_csv(RAW / "organelles" / "hpa_crosswalk.csv").itertuples(index=False, name=None)
    )
    codes = set(pd.read_csv(RAW / "organelles" / "organelles.csv")["code"])

    located, participates, has_function, encodes = [], [], [], []
    gene_props, proteins = [], []

    for e in data:
        ensg = e.get("Ensembl")
        if not ensg:
            continue
        reliability = e.get("Reliability (IF)")
        for term in as_list(e.get("Subcellular main location")):
            code = xwalk.get(term)
            if code:
                located.append(
                    {"ensembl_id": ensg, "code": code, "hpa_term": term, "reliability": reliability}
                )
        for term in as_list(e.get("Biological process")):
            participates.append({"ensembl_id": ensg, "term": term})
        for term in as_list(e.get("Molecular function")):
            has_function.append({"ensembl_id": ensg, "term": term})
        for up in as_list(e.get("Uniprot")):
            encodes.append({"ensembl_id": ensg, "uniprot_id": up})
            proteins.append({"uniprot_id": up, "name": e.get("Gene description")})
        gene_props.append(
            {
                "ensembl_id": ensg,
                "rna_tissue_specificity": e.get("RNA tissue specificity"),
                "n_interactions": e.get("Interactions"),
            }
        )

    located = pd.DataFrame(located).drop_duplicates(["ensembl_id", "code"]).reset_index(drop=True)
    participates = pd.DataFrame(participates).drop_duplicates().reset_index(drop=True)
    has_function = pd.DataFrame(has_function).drop_duplicates().reset_index(drop=True)
    encodes = pd.DataFrame(encodes).drop_duplicates().reset_index(drop=True)
    gene_props = pd.DataFrame(gene_props).drop_duplicates("ensembl_id").reset_index(drop=True)
    proteins = pd.DataFrame(proteins).drop_duplicates("uniprot_id").reset_index(drop=True)
    bp_nodes = participates[["term"]].drop_duplicates().reset_index(drop=True)
    mf_nodes = has_function[["term"]].drop_duplicates().reset_index(drop=True)

    for name, df in [
        ("hpa_located_in", located),
        ("hpa_participates", participates),
        ("hpa_has_function", has_function),
        ("hpa_encodes", encodes),
        ("hpa_gene_props", gene_props),
        ("protein_nodes", proteins),
        ("bp_nodes", bp_nodes),
        ("mf_nodes", mf_nodes),
    ]:
        df.to_parquet(INTERIM / f"{name}.parquet", index=False)

    n_ensg = gene_props["ensembl_id"].nunique()
    bad_codes = set(located["code"]) - codes
    print(f"HPA entries with ENSG: {len(gene_props)} (unique {n_ensg})")
    print(f"LOCATED_IN: {len(located)}  (genes {located['ensembl_id'].nunique()})")
    print(
        f"PARTICIPATES_IN: {len(participates)}  (genes {participates['ensembl_id'].nunique()}, BP terms {len(bp_nodes)})"
    )
    print(
        f"HAS_FUNCTION: {len(has_function)}  (genes {has_function['ensembl_id'].nunique()}, MF terms {len(mf_nodes)})"
    )
    print(f"ENCODES: {len(encodes)}  (proteins {len(proteins)})")
    print(f"reliability values: {sorted(located['reliability'].dropna().unique().tolist())}")
    assert n_ensg == len(gene_props), "ENSG not unique in HPA"
    assert not bad_codes, f"LOCATED_IN maps to non-vocabulary codes: {bad_codes}"
    assert 10000 < len(located) < 25000, f"LOCATED_IN out of expected band: {len(located)}"
    print("OK build_hpa")


if __name__ == "__main__":
    main()
