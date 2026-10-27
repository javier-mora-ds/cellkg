"""Prior exact by construction; vote back-off; no validation / test label used."""
import numpy as np

from gnn.baselines import prior_scores, vote_scores
from gnn.splits import Task


def _task() -> Task:
    edges = {"interacts": np.array([[0, 1, 3], [1, 2, 4]]), "interacts_score": np.array([900.0, 800.0, 700.0])}
    return Task("p2", "fold_1", edges, loc_train=np.array([[0, 0], [1, 1], [1, 0]]), loc_valid=np.array([[2, 1]]),
                loc_test=np.array([[4, 2]]), train_genes=np.array([0, 1]), valid_genes=np.array([2]),
                test_genes=np.array([4]), kge_source="")


def test_prior_rows_are_training_frequencies():
    P = prior_scores(_task(), 5, 3)
    assert np.allclose(P, [2 / 3, 1 / 3, 0.0])


def test_vote_uses_training_labels_only_and_backs_off():
    S, covered = vote_scores(_task(), 5, 3)
    assert covered.tolist() == [True, True, True, False, False]   # genes 3-4 have no labelled neighbour
    assert S[4, 2] < 1e-5                                         # the test label of gene 4 is never visible
    assert S[2].argmax() in (0, 1) and S[3].argmax() == 0          # back-off = prior order
