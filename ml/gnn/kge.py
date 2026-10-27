"""Re-train TransE / RotatE under Track 1's configuration, save them, export score matrices (Gate B, S1).

P1-strict replays the published partition from the persisted export and trains exactly as
``run_pykeen.py`` does (same pipeline call, seed, device, default thread count for the gate run).
P1-full and the P2 folds train on factories built from the global id maps (no coverage split).
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from gnn.common import (
    OrganelleVocab,
    ScoreMatrix,
    StageLog,
    base_parser,
    out_dir,
    resolve_args,
    update_manifest,
    write_json,
)
from gnn.metrics import positive_ranks
from gnn.pykeen_replay import ensure_export, known_locations, organelle_entities, replay
from gnn.splits import list_tasks, load_triples, p1_protocol

logging.getLogger("pykeen").setLevel(logging.ERROR)

REFERENCE = {
    "TransE": {
        "loc": {"MRR": 0.2941, "Hits@1": 0.1350, "Hits@3": 0.2750},
        "overall": {"MRR": 0.0510, "Hits@1": 0.0007, "Hits@3": 0.0720, "Hits@10": 0.1340},
    },
    "RotatE": {
        "loc": {"MRR": 0.4917, "Hits@1": 0.3400, "Hits@3": 0.5550},
        "overall": {"MRR": 0.1358, "Hits@1": 0.1025, "Hits@3": 0.1447, "Hits@10": 0.1915},
    },
}
OVERALL_KEYS = {
    "MRR": "inverse_harmonic_mean_rank",
    "Hits@1": "hits_at_1",
    "Hits@3": "hits_at_3",
    "Hits@10": "hits_at_10",
}


@dataclass(frozen=True)
class KgeJob:
    model: str
    protocol: str
    regime: str
    seed: int
    threads: int | None  # None = PyTorch default (the July condition, used for the gate run)


def _p1_replay(cfg: dict, log=print):
    df, src = ensure_export(cfg["export"]["source"], log if callable(log) else print)
    ex = cfg["export"]
    return replay(
        df,
        seed=cfg["seed"],
        interact_sample=ex["interact_sample"],
        eval_triples=ex["eval_triples"],
        loc_sample=ex["loc_sample"],
        source=src,
    )


def train_pipeline(model: str, training, testing, seed: int, cfg: dict, device: str = "cpu"):
    """The pipeline call of run_pykeen.py with the configured hyper-parameters."""
    from pykeen.pipeline import pipeline

    k = cfg["kge"]
    return pipeline(
        training=training,
        testing=testing,
        model=model,
        model_kwargs=dict(embedding_dim=k["embedding_dim"]),
        optimizer_kwargs=dict(lr=k["lr"]),
        training_kwargs=dict(num_epochs=k["epochs"], batch_size=k["batch_size"], use_tqdm=False),
        evaluation_kwargs=dict(batch_size=k["eval_batch_size"], use_tqdm=False),
        random_seed=seed,
        device=device,
    )


def score_matrix(
    model, entity_to_id: dict, relation_to_id: dict, genes: list[str], go_ids, batch: int = 2048
) -> np.ndarray:
    """S[g, o] = model.score_t([[gene, LOCATED_IN]], tails = the 22 organelles in vocabulary order)."""
    import torch

    tails = torch.tensor([entity_to_id["C:" + go] for go in go_ids], dtype=torch.long)
    r = relation_to_id["LOCATED_IN"]
    S = np.full((len(genes), len(tails)), np.nan, dtype=np.float32)
    known = [(i, entity_to_id["G:" + g]) for i, g in enumerate(genes) if "G:" + g in entity_to_id]
    model.eval()
    with torch.no_grad():
        for s in range(0, len(known), batch):
            chunk = known[s : s + batch]
            hr = torch.tensor([[e, r] for _, e in chunk], dtype=torch.long)
            S[[i for i, _ in chunk]] = model.score_t(hr, tails=tails).cpu().numpy()
    return S


def published_loop(model, res, go_ids) -> dict:
    """run_pykeen.py lines 169-189 (predict_target), per-edge ranks kept for check S1."""
    from pykeen.predict import predict_target

    orgs = organelle_entities(res.tf, go_ids)
    known = known_locations(res)
    ranks = []
    for h, _, t in res.sample200:
        pred = predict_target(
            model=model, head=h, relation="LOCATED_IN", triples_factory=res.train
        ).df
        pred = pred[pred["tail_label"].isin(orgs)]
        drop = known.get(h, set()) - {t}
        pred = (
            pred[~pred["tail_label"].isin(drop)]
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
        ranks.append(int(pred.index[pred["tail_label"] == t][0]) + 1)
    return {
        "ranks": ranks,
        "MRR": statistics.mean(1 / r for r in ranks),
        "Hits@1": statistics.mean(r <= 1 for r in ranks),
        "Hits@3": statistics.mean(r <= 3 for r in ranks),
    }


def harness_sample200(S: np.ndarray, genes: list[str], res, vocab: OrganelleVocab) -> np.ndarray:
    """Harness ranks (C22, realistic ties) for the 200 edges, one row per edge."""
    known = known_locations(res)
    gi = {g: i for i, g in enumerate(genes)}
    c2i = {go: i for i, go in enumerate(vocab.go_ids)}
    n = len(res.sample200)
    Yp = np.zeros((n, len(vocab)), dtype=bool)
    Ym = np.zeros_like(Yp)
    rows = []
    for j, (h, _, t) in enumerate(res.sample200):
        rows.append(gi[h[2:]])
        Yp[j, c2i[t[2:]]] = True
        for k in known.get(h, set()) - {t}:
            Ym[j, c2i[k[2:]]] = True
    _, _, rk = positive_ranks(S[rows], Yp, Ym, np.ones(len(vocab), dtype=bool))
    return rk


def run_kge_job(job: KgeJob, cfg: dict) -> dict:
    import torch
    from pykeen.triples import TriplesFactory

    if job.threads:
        torch.set_num_threads(job.threads)
    from gnn.graph import NodeIndex

    vocab = OrganelleVocab.load()
    genes = NodeIndex.load().genes
    res = _p1_replay(cfg)
    t0 = time.time()
    rec: dict = {
        "model": job.model,
        "protocol": job.protocol,
        "regime": job.regime,
        "seed": job.seed,
        "threads": job.threads or torch.get_num_threads(),
    }
    if job.protocol == "p1" and job.regime == "strict":
        training, testing = res.train, res.test_eval
    else:
        from gnn.splits import make_task

        task = make_task(job.protocol, job.regime)
        tr = load_triples(task.kge_source) if not task.kge_source.startswith("binary:") else None
        training = TriplesFactory.from_labeled_triples(
            tr, entity_to_id=res.tf.entity_to_id, relation_to_id=res.tf.relation_to_id
        )
        test_loc = task.loc_test[:200]
        lab = np.array(
            [["G:" + genes[g], "LOCATED_IN", "C:" + vocab.go_ids[o]] for g, o in test_loc],
            dtype=str,
        )
        testing = TriplesFactory.from_labeled_triples(
            lab, entity_to_id=res.tf.entity_to_id, relation_to_id=res.tf.relation_to_id
        )
        rec["n_training_triples"] = int(training.num_triples)
    pr = train_pipeline(job.model, training, testing, job.seed, cfg)
    rec["train_seconds"] = time.time() - t0
    protocol = p1_protocol() if job.protocol == "p1" else "p2"
    S = score_matrix(pr.model, res.tf.entity_to_id, res.tf.relation_to_id, genes, vocab.go_ids)
    tag = f"{job.model}_{protocol}_{job.regime}_seed{job.seed}"
    pr.save_to_directory(out_dir() / "models" / tag, save_training=False)
    meta = {k: v for k, v in rec.items() if k not in ("model", "protocol", "regime", "seed")}
    if job.protocol == "p1" and job.regime == "strict":
        mr = pr.metric_results
        rec["overall"] = {
            k: float(mr.get_metric(f"both.realistic.{v}")) for k, v in OVERALL_KEYS.items()
        }
        loop = published_loop(pr.model, res, vocab.go_ids)
        rk = harness_sample200(S, genes, res, vocab)
        diff = float(np.abs(1 / rk - 1 / np.array(loop["ranks"])).max())
        rec["loc_published_loop"] = {k: loop[k] for k in ("MRR", "Hits@1", "Hits@3")}
        rec["loc_harness"] = {
            "MRR": float(np.mean(1 / rk)),
            "Hits@1": float(np.mean(rk <= 1)),
            "Hits@3": float(np.mean(rk <= 3)),
        }
        rec["S1_max_abs_rr_diff"] = diff
        rec["S1_passed"] = diff <= 1e-6
        meta.update({k: rec[k] for k in ("overall", "loc_published_loop", "S1_passed")})
    ScoreMatrix(job.model, protocol, job.regime, job.seed, genes, vocab.codes, S, meta).save()
    return rec


def gate_b(records: list[dict], cfg: dict) -> dict:
    tol = cfg["kge"]["gate_b"]
    out = {"reference": REFERENCE, "tolerances": tol, "models": {}}
    for r in records:
        if not (r["protocol"] == "p1" and r["regime"] == "strict" and r["seed"] == cfg["seed"]):
            continue
        ref = REFERENCE[r["model"]]
        dloc = abs(r["loc_published_loop"]["MRR"] - ref["loc"]["MRR"])
        dov = {k: abs(r["overall"][k] - ref["overall"][k]) for k in ref["overall"]}
        tier = (
            "exact"
            if dloc <= tol["tol_exact"]
            else "accepted"
            if dloc <= tol["tol_accept"]
            else "failed"
        )
        out["models"][r["model"]] = {
            "observed_loc": r["loc_published_loop"],
            "observed_harness": r["loc_harness"],
            "observed_overall": r["overall"],
            "abs_diff_loc_MRR": dloc,
            "abs_diff_overall": dov,
            "tier": tier,
            "overall_within_tol": all(v <= tol["tol_overall"] for v in dov.values()),
            "S1_passed": r["S1_passed"],
            "S1_max_abs_rr_diff": r["S1_max_abs_rr_diff"],
        }
    rot = out["models"].get("RotatE", {}).get("tier", "missing")
    tra = out["models"].get("TransE", {})
    out["passed"] = (
        rot in ("exact", "accepted") and tra.get("abs_diff_loc_MRR", 1) <= tol["tol_accept"]
    )
    out["tier"] = rot
    return out


def main() -> None:
    ap = base_parser(
        "Re-train TransE + RotatE (identical configuration), save, score matrices, Gate B."
    )
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=3, help="torch threads per non-gate worker")
    ap.add_argument("--only-gate", action="store_true")
    args = ap.parse_args()
    cfg = resolve_args(args)
    log = StageLog("kge")
    k = cfg["kge"]
    gate_jobs = [KgeJob(m, "p1", "strict", cfg["seed"], None) for m in k["models"]]
    others = [
        KgeJob(m, "p1", "strict", s, args.threads)
        for s in k["seeds"]
        if s != cfg["seed"]
        for m in k["models"]
    ]
    others += [
        KgeJob(m, p, r, s, args.threads)
        for p, r in list_tasks(cfg)
        if not (p == "p1" and r == "strict")
        for s in k["other_seeds"]
        for m in k["models"]
    ]
    records = []
    for j in gate_jobs:  # sequential, default threads: the July condition
        records.append(run_kge_job(j, cfg))
        log(f"  {records[-1]}")
    gb = gate_b(records, cfg)
    write_json(out_dir() / "kge_reproduction.json", gb)
    log(
        f"Gate B: passed={gb['passed']} tier={gb['tier']} "
        + "; ".join(
            f"{m}: loop={v['observed_loc']} S1={v['S1_passed']}" for m, v in gb["models"].items()
        )
    )
    if not args.only_gate and others:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as ex:
            for r in ex.map(run_kge_job, others, [cfg] * len(others)):
                records.append(r)
                log(f"  {r}")
    update_manifest(
        "kge", {"gate_b": {"passed": gb["passed"], "tier": gb["tier"]}, "runs": records}, cfg
    )
    log("OK gnn.kge")


if __name__ == "__main__":
    main()
