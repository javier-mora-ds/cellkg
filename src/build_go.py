#!/usr/bin/env python3
"""Phase 2, Task 2.3 — extract the GO Cellular Component subgraph with obonet.

Keeps the `cellular_component` namespace, retains the 22 organelle anchors plus all
their ancestors, and the is_a / part_of edges among them. obonet edges point
child -> parent, so descendants() over that graph yields every ancestor of an anchor.
"""
from pathlib import Path

import networkx as nx
import obonet
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)


def main():
    g = obonet.read_obo(RAW / "go-basic.obo")
    cc_nodes = {n for n, d in g.nodes(data=True) if d.get("namespace") == "cellular_component"}
    cc = g.subgraph(cc_nodes)

    anchors = pd.read_csv(RAW / "organelles" / "organelles.csv")["go_id"].tolist()
    missing = [a for a in anchors if a not in cc]
    if missing:
        print("WARNING anchors absent from CC namespace:", missing)

    keep = set(anchors)
    for a in anchors:
        if a in cc:
            keep |= nx.descendants(cc, a)  # ancestors, via child->parent edges
    sub = cc.subgraph(keep)

    cc_rows = [
        {"go_id": n, "label": d.get("name"), "namespace": d.get("namespace")}
        for n, d in sub.nodes(data=True)
    ]
    cc_df = pd.DataFrame(cc_rows)
    cc_df.to_parquet(INTERIM / "go_cc_nodes.parquet", index=False)

    edges = []
    for child, parent, key in sub.edges(keys=True):
        if key == "is_a":
            edges.append({"child": child, "parent": parent, "rel": "SUBCLASS_OF"})
        elif key == "part_of":
            edges.append({"child": child, "parent": parent, "rel": "PART_OF"})
    edges_df = pd.DataFrame(edges)
    edges_df.to_parquet(INTERIM / "go_cc_edges.parquet", index=False)

    node_ids = set(cc_df["go_id"])
    present = [a for a in anchors if a in node_ids]
    print(
        f"CC subgraph: {len(cc_df)} nodes, {len(edges_df)} edges "
        f"({(edges_df['rel'] == 'SUBCLASS_OF').sum()} SUBCLASS_OF, "
        f"{(edges_df['rel'] == 'PART_OF').sum()} PART_OF)"
    )
    print(f"anchors present: {len(present)}/22")
    assert len(present) == 22, f"missing anchors: {set(anchors) - node_ids}"
    assert set(edges_df["child"]) <= node_ids and set(edges_df["parent"]) <= node_ids
    print("OK build_go")


if __name__ == "__main__":
    main()
