"""[slow] Gate A passed and Gate B within the accepted tier (reads reports/gnn)."""
import json

import pytest

from gnn.common import out_dir
from gnn.splits import P1_DIR, split_root

pytestmark = [pytest.mark.slow, pytest.mark.real_out]


def test_gates():
    man = split_root(P1_DIR) / "split_manifest.json"
    kr = out_dir() / "kge_reproduction.json"
    if not (man.exists() and kr.exists()):
        pytest.skip("run `make gnn-splits gnn-kge` first")
    assert json.loads(man.read_text())["gate_a"]["passed"]
    gb = json.loads(kr.read_text())
    assert gb["passed"] and gb["tier"] in ("exact", "accepted")
