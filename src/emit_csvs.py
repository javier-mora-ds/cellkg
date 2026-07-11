#!/usr/bin/env python3
"""Phase 2, Task 2.5 — emit neo4j-admin CSVs and drop isolated genes.

All cellular-component nodes share one ID space keyed by GO id; the 22 organelle
anchors carry both :CellularComponent and :Organelle labels (plus `code`), so
LOCATED_IN, ADJACENT_TO and organelle PART_OF resolve to the same nodes as the GO
is_a/part_of DAG. A gene is retained only if it has at least one biological edge
(LOCATED_IN / PARTICIPATES_IN / HAS_FUNCTION / ENCODES / INTERACTS_WITH); genes whose
only possible link is the structural CONTAINS are dropped, keeping the network dense
and median gene degree > 1.
"""
import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
IMPORT = ROOT / "data" / "import"
IMPORT.mkdir(parents=True, exist_ok=True)

CANON_CHROM = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}


def w(df, name, header):
    df.to_csv(IMPORT / name, index=False, header=header, quoting=csv.QUOTE_MINIMAL)


def main():
    genes = pd.read_parquet(INTERIM / "genes.parquet")
    valid = set(genes["ensembl_id"])
    org = pd.read_csv(RAW / "organelles" / "organelles.csv")
    code2go = dict(zip(org["code"], org["go_id"]))

    located = pd.read_parquet(INTERIM / "hpa_located_in.parquet")
    participates = pd.read_parquet(INTERIM / "hpa_participates.parquet")
    has_function = pd.read_parquet(INTERIM / "hpa_has_function.parquet")
    encodes = pd.read_parquet(INTERIM / "hpa_encodes.parquet")
    gene_props = pd.read_parquet(INTERIM / "hpa_gene_props.parquet")
    string = pd.read_parquet(INTERIM / "string_interacts.parquet")

    # restrict HPA edges to genes present in the Ensembl table
    for df in (located, participates, has_function, encodes):
        df.drop(df.index[~df["ensembl_id"].isin(valid)], inplace=True)

    # retained genes = any biological edge
    retained = (
        set(located["ensembl_id"])
        | set(participates["ensembl_id"])
        | set(has_function["ensembl_id"])
        | set(encodes["ensembl_id"])
        | set(string["source_ensg"])
        | set(string["target_ensg"])
    )
    retained &= valid

    # ---- Gene nodes ----
    g = genes[genes["ensembl_id"].isin(retained)].merge(gene_props, on="ensembl_id", how="left")
    g["n_interactions"] = g["n_interactions"].astype("Int64")
    g[":LABEL"] = "Gene;" + g["biotype_group"].fillna("OtherRNA")
    gene_cols = [
        "ensembl_id",
        "symbol",
        "biotype",
        "chromosome",
        "start",
        "end",
        "strand",
        "length",
        "rna_tissue_specificity",
        "n_interactions",
        ":LABEL",
    ]
    w(
        g[gene_cols],
        "gene_nodes.csv",
        [
            "ensembl_id:ID(Gene)",
            "symbol",
            "biotype",
            "chromosome",
            "start:long",
            "end:long",
            "strand",
            "length:long",
            "rna_tissue_specificity",
            "n_interactions:long",
            ":LABEL",
        ],
    )

    # ---- Chromosome nodes + CONTAINS ----
    contains = g[g["chromosome"].isin(CANON_CHROM)][["chromosome", "ensembl_id"]].copy()
    contains["source"] = "Ensembl"
    chrom = pd.DataFrame({"name": sorted(contains["chromosome"].unique())})
    chrom[":LABEL"] = "Chromosome"
    w(chrom, "chromosome_nodes.csv", ["name:ID(Chromosome)", ":LABEL"])
    w(contains, "contains.csv", [":START_ID(Chromosome)", ":END_ID(Gene)", "source"])

    # ---- CellularComponent (+Organelle) nodes ----
    cc = pd.read_parquet(INTERIM / "go_cc_nodes.parquet").rename(columns={"label": "go_label"})
    cc = cc.merge(
        org[["go_id", "code", "name", "description", "function", "confidence"]],
        on="go_id",
        how="left",
    )
    is_anchor = cc["code"].notna()
    cc["name"] = cc["name"].where(is_anchor, cc["go_label"])
    cc[":LABEL"] = ["CellularComponent;Organelle" if a else "CellularComponent" for a in is_anchor]
    cc_cols = [
        "go_id",
        "go_label",
        "namespace",
        "code",
        "name",
        "description",
        "function",
        "confidence",
        ":LABEL",
    ]
    w(
        cc[cc_cols],
        "cc_nodes.csv",
        [
            "go_id:ID(CC)",
            "go_label",
            "namespace",
            "code",
            "name",
            "description",
            "function",
            "confidence",
            ":LABEL",
        ],
    )

    # ---- BP / MF / Protein nodes ----
    bp = participates[["term"]].drop_duplicates().copy()
    bp[":LABEL"] = "BiologicalProcess"
    w(bp, "bp_nodes.csv", ["term:ID(BP)", ":LABEL"])
    mf = has_function[["term"]].drop_duplicates().copy()
    mf[":LABEL"] = "MolecularFunction"
    w(mf, "mf_nodes.csv", ["term:ID(MF)", ":LABEL"])
    prot_all = pd.read_parquet(INTERIM / "protein_nodes.parquet")
    prot = prot_all[prot_all["uniprot_id"].isin(set(encodes["uniprot_id"]))].copy()
    prot[":LABEL"] = "Protein"
    w(
        prot[["uniprot_id", "name", ":LABEL"]],
        "protein_nodes.csv",
        ["uniprot_id:ID(Protein)", "name", ":LABEL"],
    )

    # ---- biological relationships ----
    loc = located.copy()
    loc["code"] = loc["code"].map(code2go)  # code -> go_id (CC id space)
    loc["source"] = "HPA"
    w(
        loc[["ensembl_id", "code", "source", "reliability", "hpa_term"]],
        "located_in.csv",
        [":START_ID(Gene)", ":END_ID(CC)", "source", "reliability", "hpa_term"],
    )

    participates["source"] = "HPA"
    w(
        participates[["ensembl_id", "term", "source"]],
        "participates_in.csv",
        [":START_ID(Gene)", ":END_ID(BP)", "source"],
    )
    has_function["source"] = "HPA"
    w(
        has_function[["ensembl_id", "term", "source"]],
        "has_function.csv",
        [":START_ID(Gene)", ":END_ID(MF)", "source"],
    )
    encodes["source"] = "HPA"
    w(
        encodes[["ensembl_id", "uniprot_id", "source"]],
        "encodes.csv",
        [":START_ID(Gene)", ":END_ID(Protein)", "source"],
    )

    string = string[
        string["source_ensg"].isin(retained) & string["target_ensg"].isin(retained)
    ].copy()
    string["source"] = "STRING"
    w(
        string[["source_ensg", "target_ensg", "source", "score"]],
        "interacts_with.csv",
        [":START_ID(Gene)", ":END_ID(Gene)", "source", "score:long"],
    )

    # ---- ontology relationships (CC id space) ----
    adj = pd.read_csv(RAW / "organelles" / "organelle_adjacent_to.csv")
    adj["a"] = adj["a"].map(code2go)
    adj["b"] = adj["b"].map(code2go)
    adj["source"] = "curated"
    w(adj[["a", "b", "source"]], "adjacent_to.csv", [":START_ID(CC)", ":END_ID(CC)", "source"])

    org_part = pd.read_csv(RAW / "organelles" / "organelle_part_of.csv")
    org_part["child"] = org_part["child"].map(code2go)
    org_part["parent"] = org_part["parent"].map(code2go)
    go_edges = pd.read_parquet(INTERIM / "go_cc_edges.parquet")
    go_part = go_edges[go_edges["rel"] == "PART_OF"][["child", "parent"]]
    part = pd.concat([org_part[["child", "parent"]], go_part]).drop_duplicates().copy()
    part["source"] = "curated+GO"
    w(part, "part_of.csv", [":START_ID(CC)", ":END_ID(CC)", "source"])

    subclass = go_edges[go_edges["rel"] == "SUBCLASS_OF"][["child", "parent"]].copy()
    subclass["source"] = "GO"
    w(subclass, "subclass_of.csv", [":START_ID(CC)", ":END_ID(CC)", "source"])

    # ---- summary ----
    counts = {
        "gene_nodes": len(g),
        "chromosome_nodes": len(chrom),
        "cc_nodes": len(cc),
        "bp_nodes": len(bp),
        "mf_nodes": len(mf),
        "protein_nodes": len(prot),
        "CONTAINS": len(contains),
        "LOCATED_IN": len(loc),
        "PARTICIPATES_IN": len(participates),
        "HAS_FUNCTION": len(has_function),
        "ENCODES": len(encodes),
        "INTERACTS_WITH": len(string),
        "ADJACENT_TO": len(adj),
        "PART_OF": len(part),
        "SUBCLASS_OF": len(subclass),
    }
    total_nodes = len(g) + len(chrom) + len(cc) + len(bp) + len(mf) + len(prot)
    total_edges = sum(v for k, v in counts.items() if k[0].isupper())
    for k, v in counts.items():
        print(f"  {k:16s} {v}")
    print(f"TOTAL nodes={total_nodes}  edges={total_edges}")
    assert len(cc[cc[":LABEL"].str.contains("Organelle")]) == 22
    assert set(loc["code"]) <= set(org["go_id"]), "LOCATED_IN target outside organelle vocabulary"
    print("OK emit_csvs")


if __name__ == "__main__":
    main()
