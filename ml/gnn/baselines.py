"""Reference baselines (plan §5.G): popularity prior, neighbour vote, label propagation, feature-only MLP."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from gnn.common import (OrganelleVocab, ScoreMatrix, StageLog, base_parser, resolve_args, update_manifest)
from gnn.graph import NodeIndex
from gnn.splits import Task, list_tasks, make_task
from gnn.train import build_jobs, run_jobs

TIE_BREAK = 1e-6          # prior added to vote / propagation scores: back-off and deterministic tie-breaking


def prior_scores(task: Task, n_genes: int, n_org: int) -> np.ndarray:
    """Constant rows n_o / |E_train^LOC| (identical to the published degree baseline under P1)."""
    n = np.bincount(task.loc_train[:, 1], minlength=n_org).astype(np.float64)
    return np.tile(n / n.sum(), (n_genes, 1)).astype(np.float32)


def _adjacency(task: Task, n_genes: int) -> sp.csr_matrix:
    ei, w = task.edges["interacts"], task.edges["interacts_score"] / 1000.0
    A = sp.coo_matrix((np.r_[w, w], (np.r_[ei[0], ei[1]], np.r_[ei[1], ei[0]])), shape=(n_genes, n_genes))
    return A.tocsr()


def vote_scores(task: Task, n_genes: int, n_org: int) -> tuple[np.ndarray, float]:
    """STRING-weighted mean of the training labels of interaction neighbours; prior back-off."""
    A = _adjacency(task, n_genes)
    Y = task.labels("train", n_genes, n_org).astype(np.float64)
    labelled = Y.any(1).astype(np.float64)
    num = A @ Y
    den = A @ labelled
    S = np.divide(num, den[:, None], out=np.zeros_like(num), where=den[:, None] > 0)
    covered = den > 0
    return (S + TIE_BREAK * prior_scores(task, n_genes, n_org)).astype(np.float32), covered


def propagation_scores(task: Task, n_genes: int, n_org: int, alpha: float, sweeps: int) -> np.ndarray:
    """Zhou et al. 2004: F <- alpha D^-1/2 A D^-1/2 F + (1 - alpha) Y."""
    A = _adjacency(task, n_genes)
    d = np.asarray(A.sum(1)).ravel()
    dinv = sp.diags(np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0))
    An = dinv @ A @ dinv
    Y = task.labels("train", n_genes, n_org).astype(np.float64)
    F = Y.copy()
    for _ in range(sweeps):
        F = alpha * (An @ F) + (1 - alpha) * Y
    return (F + TIE_BREAK * prior_scores(task, n_genes, n_org)).astype(np.float32)


def main() -> None:
    ap = base_parser("Popularity prior, neighbour vote / label propagation, feature-only perceptron.")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-mlp", action="store_true")
    args = ap.parse_args()
    cfg = resolve_args(args)
    log = StageLog("baselines")
    index, vocab = NodeIndex.load(), OrganelleVocab.load()
    n_genes, n_org = len(index.genes), len(vocab)
    rec = {}
    for protocol, regime in list_tasks(cfg):
        task = make_task(protocol, regime, index, vocab)
        seed = cfg["seed"]
        P = prior_scores(task, n_genes, n_org)
        ScoreMatrix("prior", task.protocol, regime, seed, index.genes, vocab.codes, P, {}).save()
        V, cov = vote_scores(task, n_genes, n_org)
        test_cov = float(cov[task.test_genes].mean())
        ScoreMatrix("ppi_vote", task.protocol, regime, seed, index.genes, vocab.codes, V,
                    {"coverage_test": test_cov}).save()
        pc = cfg["propagation"]
        F = propagation_scores(task, n_genes, n_org, pc["alpha"], pc["sweeps"])
        ScoreMatrix("label_prop", task.protocol, regime, seed, index.genes, vocab.codes, F,
                    {"coverage_test": test_cov, **pc}).save()
        rec[f"{task.protocol}_{regime}"] = {"vote_coverage_test": test_cov}
        log(f"{task.protocol}/{regime}: prior, ppi_vote, label_prop saved (test coverage of vote {test_cov:.3f})")
    if not args.skip_mlp:
        for loss in ("bce", "logit_adjusted"):
            jobs = build_jobs(cfg, "mlp", loss, "all")
            log(f"MLP {loss}: {len(jobs)} jobs")
            rec[f"mlp_{loss}"] = run_jobs(jobs, cfg, args.jobs, log)
    update_manifest("baselines", rec, cfg)
    log("OK gnn.baselines")


if __name__ == "__main__":
    main()
