"""[slow] score_t equals predict_target on 20 genes with the saved RotatE model (reads reports/gnn)."""
import numpy as np
import pytest

from gnn.common import OrganelleVocab, load_config
from gnn.pykeen_replay import organelle_entities

pytestmark = [pytest.mark.slow, pytest.mark.real_out]


def test_score_t_equals_predict_target():
    import torch
    from pykeen.predict import predict_target

    from gnn.common import out_dir
    from gnn.kge import _p1_replay, score_matrix

    path = out_dir() / "models" / "RotatE_p1_strict_seed20260702" / "trained_model.pkl"
    if not path.exists():
        pytest.skip("run `make gnn-kge` first")
    model = torch.load(path, weights_only=False)
    res = _p1_replay(load_config())
    vocab = OrganelleVocab.load()
    genes = [h[2:] for h in res.sample200[:20, 0]]
    S = score_matrix(model, res.tf.entity_to_id, res.tf.relation_to_id, genes, vocab.go_ids)
    orgs = organelle_entities(res.tf, vocab.go_ids)
    for i, g in enumerate(genes):
        df = predict_target(model=model, head="G:" + g, relation="LOCATED_IN", triples_factory=res.train).df
        ref = df.set_index("tail_label").loc[orgs, "score"].to_numpy()
        assert np.allclose(S[i], ref, atol=1e-5)
