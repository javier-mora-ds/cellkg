"""Masked-label full-batch training on CPU (plan §5.D, §5.F) for GraphSAGE and the feature-only perceptron.

One optimizer step per epoch; early stopping on validation mAP over the supported organelles with
at least `min_valid_positives` validation positives; best checkpoint restored; the score matrix of
every gene is exported with the test-time input graph (every training localization edge visible).
"""
from __future__ import annotations

import copy
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from gnn.common import (
    OrganelleVocab,
    ScoreMatrix,
    StageLog,
    base_parser,
    out_dir,
    resolve_args,
    seed_everything,
    select_device,
    update_manifest,
)
from gnn.graph import LOC, NodeIndex, assert_no_leakage, edge_index_dict, loc_edge_index
from gnn.metrics import Harness, point
from gnn.splits import Task, list_tasks, make_task

LOSS_TAG = {
    "bce": "bce",
    "logit_adjusted": "la",
    "class_balanced": "cb",
    "asl": "asl",
    "weighted": "wbce",
    "focal": "focal",
}


@dataclass(frozen=True)
class Job:
    kind: str  # sage | mlp
    loss: str
    protocol: str  # p1 | p2
    regime: str  # strict | full | fold_k
    seed: int
    rho: float
    save_embedding: bool = False

    @property
    def model_name(self) -> str:
        return f"{self.kind}_{LOSS_TAG[self.loss]}" + ("_rho0" if self.rho == 0 else "")


@lru_cache(maxsize=8)
def _context(protocol: str, regime: str):
    from gnn.features import build_features

    index, vocab = NodeIndex.load(), OrganelleVocab.load()
    static = pd.read_parquet(out_dir() / "graph" / "features_gene.parquet")
    task = make_task(protocol, regime, index, vocab)
    X = build_features(static, task.edges, task.train_genes)
    return index, vocab, task, X, edge_index_dict(task.edges)


def _epoch_targets(
    task: Task, rng: np.random.Generator, rho: float, n_org: int, Y_train: np.ndarray
):
    """Input-part edges, loss rows, targets and loss mask for one epoch (plan §5.D)."""
    lt = task.loc_train
    if rho == 0:  # S4: every target edge also visible (leakage)
        rows = task.train_genes
        return (
            lt,
            rows,
            Y_train[rows].astype(np.float32),
            np.ones((len(rows), n_org), dtype=np.float32),
        )
    if task.protocol.startswith("p1"):
        tgt = rng.random(len(lt)) < rho
        inp, te = lt[~tgt], lt[tgt]
        rows = np.unique(te[:, 0])
        Y = np.zeros((len(rows), n_org), dtype=np.float32)
        Y[np.searchsorted(rows, te[:, 0]), te[:, 1]] = 1.0
        M = np.ones_like(Y)
        vis = inp[np.isin(inp[:, 0], rows)]
        M[
            np.searchsorted(rows, vis[:, 0]), vis[:, 1]
        ] = 0.0  # input-part positives: masked, not negative
    else:
        rows = task.train_genes[rng.random(len(task.train_genes)) < rho]
        inp = lt[~np.isin(lt[:, 0], rows)]
        Y = Y_train[rows].astype(np.float32)
        M = np.ones_like(Y)
    return inp, rows, Y, M


def run_job(job: Job, cfg: dict) -> dict:
    import torch
    from gnn.losses import ClassStats, make_loss
    from gnn.model import FeatureMLP, HeteroSAGE

    seed_everything(job.seed, cfg.get("threads"))
    device = select_device(cfg)
    index, vocab, task, X, static_eid = _context(job.protocol, job.regime)
    n_genes, n_org = len(index.genes), len(vocab)
    org_cc = index.organelle_cc_idx(vocab)
    Y_train = task.labels("train", n_genes, n_org)
    supported = Y_train[task.train_genes].any(0)
    stats = ClassStats(Y_train[task.train_genes], supported)
    loss_fn = make_loss(job.loss, stats, cfg)
    held = np.vstack([task.loc_valid, task.loc_test])
    hp = cfg["sage"] if job.kind == "sage" else cfg["mlp"]
    x = torch.as_tensor(X, device=device)
    if job.kind == "sage":
        edge_types = list(static_eid) + [LOC, ("cc", "rev_located_in", "gene")]
        model = HeteroSAGE(
            edge_types,
            index.num_nodes(),
            X.shape[1],
            d=hp["hidden"],
            L=hp["layers"],
            p=hp["dropout"],
            n_out=n_org,
        ).to(device)
        eid_static = {k: v.to(device) for k, v in static_eid.items()}
    else:
        model = FeatureMLP(X.shape[1] + n_org, hp["hidden"], hp.get("dropout", 0.3), n_org).to(
            device
        )
    opt = torch.optim.AdamW(
        model.parameters(), lr=hp["lr"], weight_decay=hp.get("weight_decay", 1e-4)
    )

    def forward(inp_edges: np.ndarray, rows: np.ndarray | None = None) -> "torch.Tensor":
        loc = loc_edge_index(inp_edges, org_cc)
        assert_no_leakage(loc[LOC], held, org_cc)  # on the exact tensor entering the model
        if job.kind == "sage":
            z = model(x, {**eid_static, **{k: v.to(device) for k, v in loc.items()}})
            return z if rows is None else z[torch.as_tensor(rows, device=device)]
        K = np.zeros((n_genes, n_org), dtype=np.float32)
        K[inp_edges[:, 0], inp_edges[:, 1]] = 1.0
        sel = slice(None) if rows is None else rows
        return model(torch.cat([x[sel], torch.as_tensor(K[sel], device=device)], 1))

    # validation set: held-out validation positives; training positives are known, not negatives
    vg = task.valid_genes
    Yv = task.labels("valid", n_genes, n_org)[vg]
    Mv = Y_train[vg] if task.protocol.startswith("p1") else np.zeros_like(Yv)
    elig = supported & (Yv.sum(0) >= cfg["metrics"]["min_valid_positives"])
    hv = Harness(
        Yv, Mv, vocab.codes, supported, cfg["metrics"]["rare_tail"], cfg["metrics"]["ultra_rare"]
    )

    rng = np.random.default_rng(job.seed)
    best, best_state, best_epoch, wait, curves = -np.inf, None, 0, 0, []
    t0 = time.time()
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        inp, rows, Y, M = _epoch_targets(task, rng, job.rho, n_org, Y_train)
        z = forward(inp, rows)
        loss = loss_fn(z, torch.as_tensor(Y, device=device), torch.as_tensor(M, device=device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["sage"]["clip"])
        opt.step()
        model.eval()
        with torch.no_grad():
            Sv = forward(task.loc_train, vg).cpu().numpy()
        m = point(hv.evaluate(Sv))
        map_elig = float(np.nanmean(np.array(m["AP_o"])[elig]))
        curves.append(
            {
                "epoch": epoch,
                "loss": loss.item(),
                "val_map14": m["mAP"],
                "val_map_elig": map_elig,
                "val_micro_mrr": m["MRR_a0@C14"],
                "val_macro_mrr": m["MRR_a1@C14"],
                "seconds": time.time() - t0,
            }
        )
        if map_elig > best:
            best, best_epoch, wait = map_elig, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if epoch >= hp.get("min_epochs", 1) and wait >= hp["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        S = forward(task.loc_train).cpu().numpy()
        emb = None
        if job.save_embedding and job.kind == "sage":
            loc = {k: v.to(device) for k, v in loc_edge_index(task.loc_train, org_cc).items()}
            emb = model.encode(x, {**eid_static, **loc})["gene"].cpu().numpy()
    meta = {
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "early_stopped": epoch < hp["epochs"],
        "best_val_map_elig": best,
        "device": device,
        "rho": job.rho,
        "loss": job.loss,
        "eligible_classes": [c for c, e in zip(vocab.codes, elig) if e],
        "leakage_assert": "passed",
        "seconds": time.time() - t0,
        "threads": cfg.get("threads"),
    }
    root = out_dir()
    sm = ScoreMatrix(
        job.model_name, task.protocol, task.regime, job.seed, index.genes, vocab.codes, S, meta
    )
    sm.save()
    if job.loss == "bce":  # zero-cost post-hoc logit adjustment
        tau = cfg["losses"]["logit_adjusted"]["tau"]
        ScoreMatrix(
            f"{job.model_name}_posthoc",
            task.protocol,
            task.regime,
            job.seed,
            index.genes,
            vocab.codes,
            S - tau * stats.lpo[None, :].astype(np.float32),
            {**meta, "posthoc_tau": tau},
        ).save()
    (root / "models").mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "job": asdict(job), "d_in": X.shape[1]},
        root / "models" / f"{sm.name}.pt",
    )
    (root / "curves").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curves).to_csv(root / "curves" / f"{sm.name}.csv", index=False)
    if emb is not None:
        np.savez_compressed(
            root / "models" / f"{sm.name}_embedding.npz",
            h=emb.astype(np.float32),
            genes=np.array(index.genes),
        )
    return {
        "name": sm.name,
        **{
            k: meta[k]
            for k in ("best_epoch", "epochs_run", "early_stopped", "best_val_map_elig", "seconds")
        },
    }


def run_jobs(jobs: list[Job], cfg: dict, n_workers: int, log: StageLog) -> list[dict]:
    results = []
    if n_workers <= 1:
        for j in jobs:
            results.append(run_job(j, cfg))
            log(f"  {results[-1]}")
        return results
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(run_job, jobs, [cfg] * len(jobs)):
            results.append(r)
            log(f"  {r}")
    return results


def build_jobs(
    cfg: dict,
    kind: str,
    loss: str,
    protocol: str,
    rho: float | None = None,
    seeds: list[int] | None = None,
    regimes: list[str] | None = None,
) -> list[Job]:
    primary = loss in ("bce", "logit_adjusted")
    seeds = seeds or (cfg["sage"]["seeds"] if primary else cfg["sage"]["ablation_seeds"])
    rho = cfg["sage"]["rho"] if rho is None else rho
    tasks = [t for t in list_tasks(cfg) if protocol == "all" or t[0] == protocol]
    if regimes:
        tasks = [t for t in tasks if t[1] in regimes]
    return [
        Job(
            kind,
            loss,
            p,
            r,
            s,
            rho,
            save_embedding=(kind == "sage" and s == seeds[0] and p == "p1"),
        )
        for p, r in tasks
        for s in seeds
    ]


def main() -> None:
    ap = base_parser("GraphSAGE masked-label training (both protocols).")
    ap.add_argument("--protocol", default="all", choices=["all", "p1", "p2"])
    ap.add_argument("--regimes", nargs="*", default=None, help="subset, e.g. full fold_1")
    ap.add_argument("--model", default="sage", choices=["sage", "mlp"])
    ap.add_argument("--loss", default="logit_adjusted", choices=list(LOSS_TAG))
    ap.add_argument(
        "--rho", type=float, default=None, help="label-mask fraction (0 = sanity check S4)"
    )
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=4, help="parallel worker processes")
    args = ap.parse_args()
    cfg = resolve_args(args)
    log = StageLog("train")
    jobs = build_jobs(cfg, args.model, args.loss, args.protocol, args.rho, args.seeds, args.regimes)
    log(
        f"{len(jobs)} jobs: model={args.model} loss={args.loss} protocol={args.protocol} rho={jobs[0].rho}"
    )
    res = run_jobs(jobs, cfg, args.jobs, log)
    update_manifest(
        f"train_{args.model}_{LOSS_TAG[args.loss]}" + ("_rho0" if args.rho == 0 else ""),
        {"jobs": res},
        cfg,
    )
    log("OK gnn.train")


if __name__ == "__main__":
    main()
