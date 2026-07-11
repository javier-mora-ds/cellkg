#!/usr/bin/env python3
"""Phase 3, Tasks 3.1 & 3.3 — apply DDL constraints/indexes and verify counts."""
from pathlib import Path

from cellkg_db import database, driver

ROOT = Path(__file__).resolve().parents[1]


def main():
    ddl = (ROOT / "cypher" / "ddl_constraints.cypher").read_text()
    clean = "\n".join(ln for ln in ddl.splitlines() if not ln.strip().startswith("//"))
    stmts = [s.strip() for s in clean.split(";") if s.strip()]

    d = driver()
    with d.session(database=database()) as s:
        for st in stmts:
            s.run(st)
        n_con = s.run("SHOW CONSTRAINTS YIELD name RETURN count(*) AS n").single()["n"]
        n_idx = s.run("SHOW INDEXES YIELD name RETURN count(*) AS n").single()["n"]
        print(f"constraints={n_con}  indexes={n_idx}")

        total_n = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
        total_r = s.run("MATCH ()-[r]->() RETURN count(r) AS n").single()["n"]
        print(f"TOTAL nodes={total_n}  relationships={total_r}")

        print("-- nodes by label --")
        for rec in s.run(
            "MATCH (n) UNWIND labels(n) AS l RETURN l AS label, count(*) AS n ORDER BY n DESC"
        ):
            print(f"  {rec['label']:20s} {rec['n']}")
        print("-- relationships by type --")
        for rec in s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n ORDER BY n DESC"):
            print(f"  {rec['t']:18s} {rec['n']}")

        org = s.run("MATCH (o:Organelle) RETURN count(o) AS n").single()["n"]
        cc = s.run("MATCH (c:CellularComponent) RETURN count(c) AS n").single()["n"]
        adj = s.run("MATCH ()-[r:ADJACENT_TO]->() RETURN count(r) AS n").single()["n"]
        badloc = s.run(
            "MATCH (:Gene)-[:LOCATED_IN]->(x) WHERE NOT x:Organelle RETURN count(*) AS n"
        ).single()["n"]
        print(
            f"Organelle={org} (expect 22)  CellularComponent={cc} (expect 52)  ADJACENT_TO={adj} (expect 43)"
        )
        print(f"LOCATED_IN targets outside Organelle vocabulary: {badloc} (expect 0)")
        assert org == 22 and cc == 52 and adj == 43 and badloc == 0
        print("OK apply_ddl_verify")
    d.close()


if __name__ == "__main__":
    main()
