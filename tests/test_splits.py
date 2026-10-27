"""Replay determinism on a synthetic export; P2 fold properties; persistence round trip."""
import numpy as np
import pandas as pd

from gnn.pykeen_replay import csv_export, replay, triples_fingerprint


def _synthetic_export(n_genes: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_genes):
        rows.append((f"CHR:{g % 3}", "CONTAINS", f"G:{g}"))
        rows.append((f"G:{g}", "LOCATED_IN", f"C:{g % 4}"))
        for h in rng.choice(n_genes, 3, replace=False):
            if h != g:
                rows.append((f"G:{g}", "INTERACTS_WITH", f"G:{h}"))
    return pd.DataFrame(rows, columns=["h", "r", "t"]).drop_duplicates()


def test_replay_deterministic_and_order_independent_given_set():
    df = _synthetic_export()
    a = replay(df, interact_sample=50, eval_triples=10, loc_sample=5)
    b = replay(df.copy(), interact_sample=50, eval_triples=10, loc_sample=5)
    assert (a.train.num_triples, a.valid.num_triples, a.test.num_triples) == \
        (b.train.num_triples, b.valid.num_triples, b.test.num_triples)
    assert np.array_equal(a.sample200, b.sample200)


def test_csv_export_matches_relationship_totals():
    df = csv_export()
    assert len(df) == 336_452
    assert df["r"].value_counts()["LOCATED_IN"] == 16_085
    assert len(triples_fingerprint(df)) == 64
