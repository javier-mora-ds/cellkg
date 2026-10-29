"""Permuted-attribute control: a seeded row permutation, its model-name suffix and its job grid (no training)."""
import numpy as np

from gnn.common import load_config
from gnn.features import permute_rows
from gnn.train import Job, build_jobs


def test_permute_rows_is_a_seeded_row_permutation_of_a_copy():
    X = np.random.default_rng(0).random((50, 4)).astype(np.float32)
    X0 = X.copy()
    P = permute_rows(X, 20260702)
    assert P.shape == X.shape and P.dtype == X.dtype
    assert not np.shares_memory(P, X) and np.array_equal(X, X0)        # a copy; the input is untouched
    assert np.array_equal(P[np.lexsort(P.T)], X[np.lexsort(X.T)])       # the same rows, reordered
    assert not np.array_equal(P, X)                                      # not the identity
    assert np.array_equal(P, permute_rows(X, 20260702))                  # reproducible per seed
    assert not np.array_equal(P, permute_rows(X, 20260703))              # different across seeds


def test_model_name_suffix_order():
    assert Job("sage", "bce", "p1", "full", 20260702, 0.5).model_name == "sage_bce"
    assert Job("sage", "bce", "p1", "full", 20260702, 0.5, features="perm").model_name == "sage_bce_xperm"
    assert Job("mlp", "bce", "p2", "fold_1", 20260702, 0.5, features="perm").model_name == "mlp_bce_xperm"
    rho0 = Job("sage", "logit_adjusted", "p1", "full", 20260702, 0.0, features="perm")
    assert rho0.model_name == "sage_la_rho0_xperm"


def test_build_jobs_passes_features_and_keeps_the_grid():
    cfg = load_config()
    real = build_jobs(cfg, "sage", "bce", "all")
    perm = build_jobs(cfg, "sage", "bce", "all", features="perm")
    assert len(real) == len(perm) == 35
    assert {j.features for j in real} == {"real"} and {j.features for j in perm} == {"perm"}
    key = lambda j: (j.protocol, j.regime, j.seed, j.save_embedding)  # noqa: E731
    assert [key(j) for j in real] == [key(j) for j in perm]
    assert {j.model_name for j in perm} == {"sage_bce_xperm"}
