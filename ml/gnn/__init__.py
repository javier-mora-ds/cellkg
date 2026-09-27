"""CellKG Track 2 — GraphSAGE embeddings and representativeness-weighted evaluation.

Additive to Track 1 (``ml/run_pykeen.py``, untouched). Every stage is a module run with
``PYTHONPATH=src:ml python -m gnn.<stage>``; artifacts go to ``reports/gnn/`` unless the
environment variable ``CELLKG_GNN_OUT`` points elsewhere (the tests use a temporary path).
"""
import os
from pathlib import Path

SEED = 20260702
ROOT = Path(__file__).resolve().parents[2]
IMPORT = ROOT / "data" / "import"
RAW = ROOT / "data" / "raw"


def out_dir() -> Path:
    """Artifact root, resolved at call time so tests can redirect it."""
    return Path(os.environ.get("CELLKG_GNN_OUT", ROOT / "reports" / "gnn"))
