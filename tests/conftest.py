"""pytest configuration: import paths and an isolated artifact directory."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "ml"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

REAL_OUT = ROOT / "reports" / "gnn"


@pytest.fixture(autouse=True)
def _isolated_out(tmp_path, monkeypatch, request):
    """Tests never write into reports/: CELLKG_GNN_OUT -> tmp_path (tests marked `real_out` read the real one)."""
    if "real_out" not in request.keywords:
        monkeypatch.setenv("CELLKG_GNN_OUT", str(tmp_path))
