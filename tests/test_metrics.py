"""Harness identities on the toy matrix of plan Appendix B.5, and parity with scikit-learn."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from gnn.metrics import Harness, ap_auc, bootstrap_weights, fit_thresholds, point, skill

S = np.array([[0.9, 0.1, 0.5, 0.2], [0.3, 0.8, 0.2, 0.7], [0.4, 0.4, 0.9, 0.1]])
Y = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 1, 0]], dtype=bool)
M = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool)
CODES = ("A", "B", "C", "D")


def harness() -> Harness:
    return Harness(Y, M, CODES, np.ones(4, dtype=bool), ["A"], ["A"], ks=(1, 3), alphas=(0.0, 0.5, 1.0))


def test_toy_rank_identities():
    m = point(harness().evaluate(S))
    assert m["MRR_a0@C14"] == pytest.approx(0.75)
    assert m["Hits@1_a0@C14"] == pytest.approx(0.5)
    assert m["Hits@3_a0@C14"] == pytest.approx(1.0)
    assert m["MRR_a1@C14"] == pytest.approx(2.0 / 3.0, abs=1e-4)          # macro over classes with positives
    mrr_o = np.array(m["MRR_o"])
    sup = np.array(m["support_o"])
    micro = (sup * np.nan_to_num(mrr_o)).sum() / sup.sum()
    assert m["MRR_a0@C14"] == pytest.approx(micro)
    assert np.isnan(mrr_o[3]) and np.isnan(m["AP_o"][3]) and np.isnan(m["AUROC_o"][3])


def test_skill_score():
    assert skill(np.array(0.75), np.array(0.5)) == pytest.approx(0.5)


def test_ap_auc_matches_sklearn_with_weights_and_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 300
        y = rng.random(n) < 0.1
        s = np.round(rng.normal(size=n) + y, 1)                          # rounding creates ties
        w = rng.integers(0, 3, n).astype(float)
        if not (w * y).sum() or not (w * ~y).sum():
            continue
        ap, auc = ap_auc(y, s, w[None, :])
        assert ap[0] == pytest.approx(average_precision_score(y, s, sample_weight=w), abs=1e-10)
        assert auc[0] == pytest.approx(roc_auc_score(y, s, sample_weight=w), abs=1e-10)


def test_constant_column_is_prevalence_and_half():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    ap, auc = ap_auc(y, np.full(10, 0.3), np.ones((1, 10)))
    assert ap[0] == pytest.approx(0.2) and auc[0] == pytest.approx(0.5)


def test_bootstrap_rows_sum_to_n_and_reproducible():
    W = bootstrap_weights(50, 7, 1)
    assert (W.sum(1) == 50).all() and np.array_equal(W, bootstrap_weights(50, 7, 1))
    out = harness().evaluate(S, W=bootstrap_weights(3, 5, 2))
    assert out["MRR_a0@C14"].shape == (5,)


def test_thresholds_use_validation_only_and_never_fire_unsupported():
    theta = fit_thresholds(S, Y, M, np.array([1, 1, 1, 0], dtype=bool), min_pos=1)
    assert np.isinf(theta[3]) and np.isfinite(theta[:3]).all()
