"""Exact replay of Track 1's export -> cap -> split -> two rng draws, and Gate A.

`export_triples` and `EXCLUDE_REL` are imported from the untouched ``ml/run_pykeen.py``; the
deterministic steps living inside its ``main()`` (lines 73-98 and 131-167) are re-implemented
verbatim here. ``tests/test_replay_drift.py`` pins the SHA-256 of the original file.
"""
from __future__ import annotations

import hashlib
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from gnn import IMPORT, ROOT
from gnn.common import StageLog, out_dir, write_json

logging.getLogger("pykeen.triples").setLevel(logging.ERROR)  # silence label-reconstruction notices

RUN_PYKEEN = ROOT / "ml" / "run_pykeen.py"
REFERENCE_SIZES = (97_133, 12_142, 12_142)
REFERENCE_COUNTS = {"triples": 121_417, "entities": 20_140, "relations": 8}
REFERENCE_DEGREE = {"MRR": 0.5936, "Hits@1": 0.4400, "Hits@3": 0.6600}
REFERENCE_RANDOM = 0.2369

# Relationship files of data/import/ and their Neo4j type, with the entity prefixes of run_pykeen.
CSV_RELATIONS = [
    ("located_in.csv", "LOCATED_IN", "G:", "C:"),
    ("interacts_with.csv", "INTERACTS_WITH", "G:", "G:"),
    ("participates_in.csv", "PARTICIPATES_IN", "G:", "BP:"),
    ("has_function.csv", "HAS_FUNCTION", "G:", "MF:"),
    ("contains.csv", "CONTAINS", "CHR:", "G:"),
    ("subclass_of.csv", "SUBCLASS_OF", "C:", "C:"),
    ("part_of.csv", "PART_OF", "C:", "C:"),
    ("adjacent_to.csv", "ADJACENT_TO", "C:", "C:"),
    ("encodes.csv", "ENCODES", "G:", "P:"),
]


@dataclass
class ReplayResult:
    tf: object
    train: object
    valid: object
    test: object
    test_eval: object  # the 3,000-triple subsample (overall metrics)
    sample200: np.ndarray  # labelled (h, r, t) rows
    export_source: str


def export_path(source: str) -> Path:
    return out_dir() / "export" / f"pykeen_export_{source}.parquet"


def csv_export() -> pd.DataFrame:
    """The same labelled triples as run_pykeen.export_triples, assembled from data/import/."""
    frames = []
    for fname, rel, hp, tp in CSV_RELATIONS:
        d = pd.read_csv(IMPORT / fname, dtype=str, usecols=[0, 1])
        d.columns = ["h", "t"]
        frames.append(pd.DataFrame({"h": hp + d["h"], "r": rel, "t": tp + d["t"]}))
    return pd.concat(frames, ignore_index=True)


def triples_fingerprint(df: pd.DataFrame) -> str:
    s = (
        df[["h", "r", "t"]]
        .astype(str)
        .sort_values(["h", "r", "t"])
        .to_csv(index=False, header=False)
    )
    return hashlib.sha256(s.encode()).hexdigest()


def order_fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        df[["h", "r", "t"]].to_csv(index=False, header=False).encode()
    ).hexdigest()


def ensure_export(source: str, log: StageLog) -> tuple[pd.DataFrame, str]:
    """Return the export (h, r, t) and its source label, persisting it on first use.

    source = auto: persisted Neo4j export if present, else live Neo4j (persisted), else CSV fallback.
    """
    neo = export_path("neo4j")
    if source in ("auto", "neo4j") and neo.exists():
        log(f"export: persisted Neo4j export {neo.relative_to(out_dir())}")
        return pd.read_parquet(neo), "neo4j"
    if source in ("auto", "neo4j"):
        try:
            from run_pykeen import export_triples  # untouched module

            df = export_triples()
            neo.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(neo, index=False)
            log(f"export: live Neo4j export persisted ({len(df)} rows)")
            return df, "neo4j"
        except Exception as exc:  # noqa: BLE001
            if source == "neo4j":
                raise
            log(f"export: Neo4j unavailable ({type(exc).__name__}); CSV fallback -> protocol p1r")
    df = csv_export()
    p = export_path("csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df, "csv"


def export_parity(log: StageLog) -> dict:
    """Compare the persisted Neo4j export with the CSV-order export (set and order equality)."""
    csv = csv_export()
    csv.to_parquet(export_path("csv"), index=False)
    out = {"csv_rows": len(csv), "csv_set_sha256": triples_fingerprint(csv)}
    neo = export_path("neo4j")
    if neo.exists():
        n = pd.read_parquet(neo)
        out.update(
            {
                "neo4j_rows": len(n),
                "neo4j_set_sha256": triples_fingerprint(n),
                "set_equal": triples_fingerprint(n) == out["csv_set_sha256"],
                "order_equal": order_fingerprint(n) == order_fingerprint(csv),
            }
        )
    write_json(out_dir() / "export" / "parity.json", out)
    log(f"export parity: {out}")
    return out


def replay(
    df: pd.DataFrame,
    *,
    seed: int = 20260702,
    interact_sample: int = 40_000,
    eval_triples: int = 3_000,
    loc_sample: int = 200,
    source: str = "neo4j",
) -> ReplayResult:
    from pykeen.triples import TriplesFactory
    from run_pykeen import EXCLUDE_REL

    df = df[~df["r"].isin(EXCLUDE_REL)]
    inter = df[df["r"] == "INTERACTS_WITH"]
    rest = df[df["r"] != "INTERACTS_WITH"]
    if len(inter) > interact_sample:
        inter = inter.sample(interact_sample, random_state=seed)
    df = pd.concat([rest, inter], ignore_index=True)
    tf = TriplesFactory.from_labeled_triples(df[["h", "r", "t"]].to_numpy(dtype=str))
    train, valid, test = tf.split([0.8, 0.1, 0.1], random_state=seed)
    rng = np.random.default_rng(seed)
    idx = rng.choice(test.num_triples, min(eval_triples, test.num_triples), replace=False)
    test_eval = TriplesFactory(
        mapped_triples=test.mapped_triples[idx],
        entity_to_id=tf.entity_to_id,
        relation_to_id=tf.relation_to_id,
    )
    test_loc = test.triples[test.triples[:, 1] == "LOCATED_IN"]
    sample = test_loc
    if len(sample) > loc_sample:
        sample = sample[rng.choice(len(sample), loc_sample, replace=False)]
    return ReplayResult(tf, train, valid, test, test_eval, sample, source)


def organelle_entities(tf, go_ids) -> list[str]:
    """run_pykeen's ``orgs``: organelle entities in organelles.csv order present in the id map."""
    return [o for o in ("C:" + pd.Series(list(go_ids))) if o in tf.entity_to_id]


def known_locations(res: ReplayResult) -> dict[str, set[str]]:
    known: dict[str, set[str]] = {}
    for arr in (res.train.triples, res.valid.triples, res.test.triples):
        for h, _r, t in arr[arr[:, 1] == "LOCATED_IN"]:
            known.setdefault(h, set()).add(t)
    return known


def degree_baseline_exact(res: ReplayResult, go_ids) -> dict:
    """run_pykeen.py lines 131-167, verbatim: random and degree baselines on sample200."""
    orgs = organelle_entities(res.tf, go_ids)
    train_loc = res.train.triples[res.train.triples[:, 1] == "LOCATED_IN"]
    pop = pd.Series(train_loc[:, 2]).value_counts()
    pop_order = [o for o in pop.index if o in orgs]
    known = known_locations(res)
    rr_deg, h1_deg, h3_deg, rr_rand = [], [], [], []
    for h, _, t in res.sample200:
        cand = [o for o in pop_order if o == t or o not in known.get(h, set())]
        if t not in cand:
            cand.append(t)
        rank = cand.index(t) + 1
        rr_deg.append(1 / rank)
        h1_deg.append(rank <= 1)
        h3_deg.append(rank <= 3)
        c = len(cand)
        rr_rand.append(sum(1 / k for k in range(1, c + 1)) / c)
    return {
        "MRR": statistics.mean(rr_deg),
        "Hits@1": statistics.mean(h1_deg),
        "Hits@3": statistics.mean(h3_deg),
        "random_MRR": statistics.mean(rr_rand),
        "n_orgs": len(orgs),
        "n_pop_order": len(pop_order),
    }


def gate_a(res: ReplayResult, go_ids, tol: float = 5e-5) -> dict:
    observed = degree_baseline_exact(res, go_ids)
    sizes = (res.train.num_triples, res.valid.num_triples, res.test.num_triples)
    counts = {
        "triples": int(res.tf.num_triples),
        "entities": int(res.tf.num_entities),
        "relations": int(res.tf.num_relations),
    }
    passed = (
        sizes == REFERENCE_SIZES
        and counts == REFERENCE_COUNTS
        and all(abs(observed[k] - REFERENCE_DEGREE[k]) <= tol for k in REFERENCE_DEGREE)
    )
    return {
        "passed": passed,
        "protocol": "p1" if passed else "p1r",
        "observed": observed,
        "reference": REFERENCE_DEGREE,
        "reference_random": REFERENCE_RANDOM,
        "sizes": sizes,
        "reference_sizes": REFERENCE_SIZES,
        "counts": counts,
        "export_source": res.export_source,
        "tolerance": tol,
    }
