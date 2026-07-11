#!/usr/bin/env python3
"""Phase 4 — graph characterization and interesting-network acceptance gates.

Computes label/type counts, connected-component structure, gene-degree distribution,
the circuit rank (independent cycles) of the largest component, the depth of the GO
Cellular Component hierarchy, and biotype-per-organelle composition. Enforces the
gates: LCC >= 95% of nodes, cycles present, median gene degree > 1, and every
LOCATED_IN target within the 22-organelle vocabulary.
"""
import statistics
from pathlib import Path

import networkx as nx
from cellkg_db import database, driver

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "characterization.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    d = driver()
    with d.session(database=database()) as s:
        nodes = {
            r["id"]: r["labels"]
            for r in s.run("MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels")
        }
        G = nx.Graph()
        G.add_nodes_from(nodes)
        for r in s.run("MATCH (a)-->(b) RETURN elementId(a) AS a, elementId(b) AS b"):
            G.add_edge(r["a"], r["b"])

        emit("=== CellKG characterization ===")
        emit(f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

        comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
        lcc = comps[0]
        lcc_frac = lcc / G.number_of_nodes()
        emit(
            f"connected components={len(comps)}  LCC={lcc} ({lcc_frac:.4f})  next largest={comps[1] if len(comps) > 1 else 0}"
        )

        gene_ids = [i for i, l in nodes.items() if "Gene" in l]
        gdeg = [G.degree(i) for i in gene_ids]
        emit(
            f"gene degree: n={len(gdeg)} min={min(gdeg)} median={statistics.median(gdeg)} "
            f"mean={statistics.mean(gdeg):.2f} max={max(gdeg)}"
        )

        lcc_nodes = max(nx.connected_components(G), key=len)
        H = G.subgraph(lcc_nodes)
        circuit_rank = H.number_of_edges() - H.number_of_nodes() + 1
        emit(
            f"LCC circuit rank (independent cycles)={circuit_rank}  -> cycles present: {circuit_rank > 0}"
        )

        D = nx.DiGraph()
        for r in s.run(
            "MATCH (a:CellularComponent)-[:SUBCLASS_OF|PART_OF]->(b:CellularComponent) "
            "RETURN elementId(a) AS a, elementId(b) AS b"
        ):
            D.add_edge(r["a"], r["b"])
        is_dag = nx.is_directed_acyclic_graph(D)
        depth = nx.dag_longest_path_length(D) if is_dag and D.number_of_nodes() else -1
        emit(
            f"GO CC hierarchy: nodes={D.number_of_nodes()} edges={D.number_of_edges()} "
            f"DAG={is_dag} longest_path_depth={depth}"
        )

        emit("-- genes localized per organelle (top 12) --")
        for r in s.run(
            "MATCH (g:Gene)-[:LOCATED_IN]->(o:Organelle) "
            "RETURN o.code AS code, o.name AS name, count(g) AS genes "
            "ORDER BY genes DESC LIMIT 12"
        ):
            emit(f"  {r['code']:4s} {r['name'][:26]:26s} {r['genes']}")

        emit("-- gene biotype composition (retained graph) --")
        for r in s.run(
            "MATCH (g:Gene) RETURN g.biotype AS biotype, count(*) AS n ORDER BY n DESC LIMIT 12"
        ):
            emit(f"  {str(r['biotype']):22s} {r['n']}")

        badloc = s.run(
            "MATCH (:Gene)-[:LOCATED_IN]->(x) WHERE NOT x:Organelle " "RETURN count(*) AS n"
        ).single()["n"]

    OUT.write_text("\n".join(lines) + "\n")
    emit(f"\nreport written to {OUT.relative_to(ROOT)}")

    # ---- acceptance gates ----
    assert lcc_frac >= 0.95, f"LCC {lcc_frac:.3f} < 0.95"
    assert circuit_rank > 0, "no cycles"
    assert statistics.median(gdeg) > 1, "median gene degree not > 1"
    assert badloc == 0, "LOCATED_IN outside organelle vocabulary"
    emit("ALL GATES PASSED")
    d.close()


if __name__ == "__main__":
    main()
