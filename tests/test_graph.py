"""Graph assembly from data/import/: counts, symmetric interaction layer, leakage assertion."""
import numpy as np
import pytest

from gnn.common import OrganelleVocab
from gnn.graph import LOC, NodeIndex, assert_no_leakage, csv_edges, edge_index_dict, loc_edge_index


@pytest.fixture(scope="module")
def built():
    index = NodeIndex.build()
    return index, csv_edges(index), OrganelleVocab.load()


def test_node_counts(built):
    index, _, vocab = built
    assert index.num_nodes() == {"gene": 19_702, "cc": 52, "bp": 232, "mf": 129, "chromosome": 25}
    assert len(vocab) == 22 and len(set(index.organelle_cc_idx(vocab))) == 22


def test_interaction_layer_symmetric(built):
    _, full, _ = built
    ei = edge_index_dict(full)[("gene", "interacts", "gene")].numpy()
    assert ei.shape[1] == 471_366
    fwd = set(map(tuple, ei.T.tolist()))
    assert all((b, a) in fwd for a, b in fwd)
    assert not any(a == b for a, b in fwd)


def test_leakage_assertion_raises_on_planted_edge(built):
    index, _, vocab = built
    org_cc = index.organelle_cc_idx(vocab)
    train = np.array([[0, 1], [5, 2]])
    held = np.array([[7, 3]])
    assert_no_leakage(loc_edge_index(train, org_cc)[LOC], held, org_cc)
    with pytest.raises(AssertionError):
        assert_no_leakage(loc_edge_index(np.vstack([train, held]), org_cc)[LOC], held, org_cc)
