#!/usr/bin/env python3
"""Phase 6 — PyKEEN link prediction (TransE + RotatE) for gene->organelle localization.

Exports triples from Neo4j, trains TransE and RotatE (fixed seed), reports overall
filtered MRR / Hits@{1,3,10} on a held-out test subsample, and evaluates the primary
objective (predicting LOCATED_IN) restricted to the 22 organelle tails against a random
and a degree (popularity) baseline. Finally it emits ranked novel gene->organelle
localizations for a panel of well-known genes.

The protein/ENCODES layer is excluded from the embedding graph (proteins do not inform
localization and would double the entity space), and the dominant INTERACTS_WITH layer
is down-sampled, so the whole stage runs in a few minutes on CPU.
"""
import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cellkg_db import database, driver  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "ml_results.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)
EXCLUDE_REL = {"ENCODES"}

ENTITY_CASE = (
    "CASE WHEN {v}:Gene THEN 'G:'+{v}.ensembl_id "
    "WHEN {v}:CellularComponent THEN 'C:'+{v}.go_id "
    "WHEN {v}:Chromosome THEN 'CHR:'+{v}.name "
    "WHEN {v}:BiologicalProcess THEN 'BP:'+{v}.term "
    "WHEN {v}:MolecularFunction THEN 'MF:'+{v}.term "
    "WHEN {v}:Protein THEN 'P:'+{v}.uniprot_id END"
)


def export_triples():
    q = (
        f"MATCH (h)-[r]->(t) RETURN {ENTITY_CASE.format(v='h')} AS h, "
        f"type(r) AS r, {ENTITY_CASE.format(v='t')} AS t"
    )
    d = driver()
    with d.session(database=database()) as s:
        rows = [(rec["h"], rec["r"], rec["t"]) for rec in s.run(q)]
    d.close()
    return pd.DataFrame(rows, columns=["h", "r", "t"]).dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--interact-sample", type=int, default=40000)
    ap.add_argument("--eval-triples", type=int, default=3000)
    ap.add_argument("--loc-sample", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from pykeen.pipeline import pipeline
    from pykeen.predict import predict_target
    from pykeen.triples import TriplesFactory

    log = []

    def emit(s=""):
        print(s, flush=True)
        log.append(s)

    df = export_triples()
    df = df[~df["r"].isin(EXCLUDE_REL)]
    inter = df[df["r"] == "INTERACTS_WITH"]
    rest = df[df["r"] != "INTERACTS_WITH"]
    if len(inter) > args.interact_sample:
        inter = inter.sample(args.interact_sample, random_state=args.seed)
    df = pd.concat([rest, inter], ignore_index=True)
    emit(
        f"training on {len(df)} triples (excluded {sorted(EXCLUDE_REL)}, "
        f"INTERACTS_WITH capped at {args.interact_sample})"
    )

    tf = TriplesFactory.from_labeled_triples(df[["h", "r", "t"]].to_numpy(dtype=str))
    train, valid, test = tf.split([0.8, 0.1, 0.1], random_state=args.seed)
    emit(
        f"entities={tf.num_entities} relations={tf.num_relations} "
        f"train/valid/test={train.num_triples}/{valid.num_triples}/{test.num_triples}"
    )

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(test.num_triples, min(args.eval_triples, test.num_triples), replace=False)
    test_eval = TriplesFactory(
        mapped_triples=test.mapped_triples[idx],
        entity_to_id=tf.entity_to_id,
        relation_to_id=tf.relation_to_id,
    )

    models = {}
    for model in ["TransE", "RotatE"]:
        emit(f"\n=== training {model} (dim={args.dim}, epochs={args.epochs}) ===")
        res = pipeline(
            training=train,
            testing=test_eval,
            model=model,
            model_kwargs=dict(embedding_dim=args.dim),
            optimizer_kwargs=dict(lr=0.005),
            training_kwargs=dict(num_epochs=args.epochs, batch_size=2048, use_tqdm=False),
            evaluation_kwargs=dict(batch_size=512, use_tqdm=False),
            random_seed=args.seed,
            device=args.device,
        )
        mr = res.metric_results
        m = {
            k: mr.get_metric(f"both.realistic.{v}")
            for k, v in {
                "MRR": "inverse_harmonic_mean_rank",
                "Hits@1": "hits_at_1",
                "Hits@3": "hits_at_3",
                "Hits@10": "hits_at_10",
            }.items()
        }
        models[model] = res.model
        emit(
            f"{model} overall (all-entity filtered, {test_eval.num_triples} test triples): "
            + "  ".join(f"{k}={v:.4f}" for k, v in m.items())
        )

    # ---- localization objective: LOCATED_IN restricted to 22 organelle tails ----
    org = pd.read_csv(ROOT / "data/raw/organelles/organelles.csv")
    orgs = [o for o in ("C:" + org["go_id"]) if o in tf.entity_to_id]
    code_by_go = dict(zip("C:" + org["go_id"], org["code"]))

    train_loc = train.triples[train.triples[:, 1] == "LOCATED_IN"]
    pop = pd.Series(train_loc[:, 2]).value_counts()
    pop_order = [o for o in pop.index if o in orgs]

    known = {}
    for arr in (train.triples, valid.triples, test.triples):
        for h, r, t in arr[arr[:, 1] == "LOCATED_IN"]:
            known.setdefault(h, set()).add(t)

    test_loc = test.triples[test.triples[:, 1] == "LOCATED_IN"]
    sample = test_loc
    if len(sample) > args.loc_sample:
        sample = sample[rng.choice(len(sample), args.loc_sample, replace=False)]
    emit(
        f"\n=== localization link prediction (LOCATED_IN, {len(sample)} test edges, {len(orgs)} organelle tails) ==="
    )

    rr_deg, h1_deg, h3_deg, rr_rand = [], [], [], []
    for h, _, t in sample:
        cand = [o for o in pop_order if o == t or o not in known.get(h, set())]
        if t not in cand:
            cand.append(t)
        rank = cand.index(t) + 1
        rr_deg.append(1 / rank)
        h1_deg.append(rank <= 1)
        h3_deg.append(rank <= 3)
        c = len(cand)
        rr_rand.append(sum(1 / k for k in range(1, c + 1)) / c)
    emit(f"random baseline:  MRR={statistics.mean(rr_rand):.4f}")
    emit(
        f"degree baseline:  MRR={statistics.mean(rr_deg):.4f}  "
        f"Hits@1={statistics.mean(h1_deg):.4f}  Hits@3={statistics.mean(h3_deg):.4f}"
    )

    for name, model in models.items():
        rr, h1, h3 = [], [], []
        for h, _, t in sample:
            pred = predict_target(
                model=model, head=h, relation="LOCATED_IN", triples_factory=train
            ).df
            pred = pred[pred["tail_label"].isin(orgs)]
            drop = known.get(h, set()) - {t}
            pred = (
                pred[~pred["tail_label"].isin(drop)]
                .sort_values("score", ascending=False)
                .reset_index(drop=True)
            )
            rank = int(pred.index[pred["tail_label"] == t][0]) + 1
            rr.append(1 / rank)
            h1.append(rank <= 1)
            h3.append(rank <= 3)
        emit(
            f"{name:7s} localization: MRR={statistics.mean(rr):.4f}  "
            f"Hits@1={statistics.mean(h1):.4f}  Hits@3={statistics.mean(h3):.4f}"
        )

    # ---- ranked novel gene->organelle predictions (RotatE) ----
    emit("\n=== ranked novel gene->organelle predictions (RotatE) ===")
    best = models["RotatE"]
    panel = ["TP53", "BRCA1", "EGFR", "MYC", "VDAC1", "GAPDH", "LMNA", "CANX"]
    d = driver()
    with d.session(database=database()) as s:
        sym2ensg = {
            r["s"]: "G:" + r["e"]
            for r in s.run(
                "MATCH (g:Gene) WHERE g.symbol IN $p RETURN g.symbol AS s, g.ensembl_id AS e",
                p=panel,
            )
        }
    d.close()
    for sym in panel:
        h = sym2ensg.get(sym)
        if not h or h not in tf.entity_to_id:
            continue
        pred = predict_target(model=best, head=h, relation="LOCATED_IN", triples_factory=train).df
        pred = pred[pred["tail_label"].isin(orgs)].sort_values("score", ascending=False)
        known_h = known.get(h, set())
        novel = [
            (code_by_go.get(r.tail_label, r.tail_label), round(float(r.score), 2))
            for r in pred.itertuples()
            if r.tail_label not in known_h
        ][:3]
        kn = sorted(code_by_go.get(x, x) for x in known_h)
        emit(f"  {sym:6s} known={kn}  top-novel={novel}")

    OUT.write_text("\n".join(log) + "\n")
    emit(f"\nreport written to {OUT.relative_to(ROOT)}")
    emit("OK run_pykeen")


if __name__ == "__main__":
    main()
