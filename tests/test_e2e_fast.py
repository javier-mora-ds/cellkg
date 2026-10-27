"""[slow] `make gnn-fast` into a temporary artifact directory produces every table and score matrix."""
import os
import shutil
import subprocess

import pytest

from gnn import ROOT

pytestmark = pytest.mark.slow


def test_make_gnn_fast(tmp_path):
    real = ROOT / "reports" / "gnn" / "export"
    if (real / "pykeen_export_neo4j.parquet").exists():          # exact replay offline
        shutil.copytree(real, tmp_path / "export")
    env = {**os.environ, "CELLKG_GNN_OUT": str(tmp_path)}
    subprocess.run(["make", "gnn-fast"], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    assert (tmp_path / "comparison_table.md").exists()
    names = {p.name.split("_seed")[0] for p in (tmp_path / "scores").glob("*.npz")}
    for m in ("prior", "ppi_vote", "label_prop", "TransE", "RotatE", "mlp_bce", "mlp_la", "sage_bce", "sage_la"):
        assert any(n.startswith(m + "_") for n in names), m
