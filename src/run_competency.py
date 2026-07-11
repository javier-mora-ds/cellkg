#!/usr/bin/env python3
"""Phase 5 — run the competency questions and capture results to reports/."""
import re
from pathlib import Path

from cellkg_db import database, driver

ROOT = Path(__file__).resolve().parents[1]
CQL = ROOT / "cypher" / "competency_all.cypher"
OUT = ROOT / "reports" / "competency_results.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)

MARK = re.compile(r"^// @@ \| (Q\d+) \| (.*)$")


def parse_blocks(text):
    blocks, cur = [], None
    for line in text.splitlines():
        m = MARK.match(line)
        if m:
            cur = {"id": m.group(1), "title": m.group(2), "lines": []}
            blocks.append(cur)
        elif cur is not None and not line.strip().startswith("//"):
            cur["lines"].append(line)
    for b in blocks:
        b["cypher"] = "\n".join(b["lines"]).strip().rstrip(";").strip()
    return blocks


def fmt_rows(rows, limit=15):
    out = []
    for rec in rows[:limit]:
        out.append("    " + " | ".join(f"{k}={rec[k]}" for k in rec.keys()))
    return out


def main():
    blocks = parse_blocks(CQL.read_text())
    d = driver()
    report, empties = [], []
    with d.session(database=database()) as s:
        for b in blocks:
            report.append(f"\n### {b['id']}: {b['title']}")
            try:
                rows = list(s.run(b["cypher"]))
            except Exception as e:
                report.append(f"    ERROR: {e}")
                empties.append(b["id"])
                continue
            report.append(f"    ({len(rows)} rows)")
            report.extend(fmt_rows(rows))
            if not rows:
                empties.append(b["id"])
    d.close()

    text = "=== CellKG competency question results ===\n" + "\n".join(report) + "\n"
    OUT.write_text(text)
    print(text)
    print(f"report written to {OUT.relative_to(ROOT)}")
    if empties:
        print(f"WARNING empty/failed queries: {empties}")
    else:
        print("OK all 10 competency questions returned non-empty results")


if __name__ == "__main__":
    main()
