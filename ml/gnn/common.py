"""Shared utilities: configuration, seeding, device, manifest, vocabulary, score matrices, I/O."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from gnn import IMPORT, RAW, ROOT, SEED, out_dir

CONFIG_DIR = Path(__file__).resolve().parent / "config"


# ---------------------------------------------------------------- configuration
def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None, fast: bool = False) -> dict:
    cfg = yaml.safe_load(Path(path or CONFIG_DIR / "default.yaml").read_text())
    if fast:
        cfg = _merge(cfg, yaml.safe_load((CONFIG_DIR / "fast.yaml").read_text()))
    cfg["fast"] = bool(fast)
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default=str(CONFIG_DIR / "default.yaml"))
    ap.add_argument("--fast", action="store_true", help="smoke run (config/fast.yaml overrides)")
    ap.add_argument("--seed", type=int, default=None, help="override the base seed")
    ap.add_argument("--device", default=None, help="cpu | mps (reported numbers are CPU-only)")
    ap.add_argument(
        "--out", default=None, help="artifact root (default reports/gnn or $CELLKG_GNN_OUT)"
    )
    return ap


def resolve_args(args: argparse.Namespace) -> dict:
    if args.out:
        os.environ["CELLKG_GNN_OUT"] = str(Path(args.out).expanduser().resolve())
    cfg = load_config(args.config, args.fast)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device:
        cfg["device"] = args.device
    return cfg


# ---------------------------------------------------------------- seeding / device
def seed_everything(seed: int, threads: int | None = None) -> None:
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if threads:
        torch.set_num_threads(int(threads))
    torch.use_deterministic_algorithms(True)


def select_device(cfg: dict) -> str:
    import torch

    want = os.environ.get("CELLKG_GNN_DEVICE", cfg.get("device", "cpu"))
    if want == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return want


# ---------------------------------------------------------------- logging / manifest
class StageLog:
    """Print and append to reports/gnn/logs/<stage>.log."""

    def __init__(self, stage: str):
        self.stage = stage
        self.path = out_dir() / "logs" / f"{stage}.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self("--- start " + time.strftime("%Y-%m-%d %H:%M:%S"))

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        with self.path.open("a") as fh:
            fh.write(msg + "\n")

    def elapsed(self) -> float:
        return time.time() - self.t0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions() -> dict:
    import importlib.metadata as md

    out = {"python": sys.version.split()[0]}
    for pkg in (
        "torch",
        "torch_geometric",
        "pykeen",
        "numpy",
        "pandas",
        "scikit-learn",
        "iterative-stratification",
        "matplotlib",
    ):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def _git_state() -> dict:
    try:
        head = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True
            ).stdout.strip()
        )
        return {"head": head, "dirty": dirty}
    except Exception:  # noqa: BLE001 — git absent is not an error
        return {"head": None, "dirty": None}


def data_fingerprints() -> dict:
    return {p.name: sha256_file(p) for p in sorted(IMPORT.glob("*.csv"))}


def update_manifest(stage: str, record: dict, cfg: dict | None = None) -> None:
    """Merge one stage record into reports/gnn/run_manifest.json under an exclusive file lock."""
    path = out_dir() / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "versions": _versions(),
        "platform": platform.platform(),
        "git": _git_state(),
        **record,
    }
    if cfg is not None:
        rec["config_hash"] = config_hash(cfg)
        rec["fast"] = cfg.get("fast", False)
    with open(path.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        man = json.loads(path.read_text()) if path.exists() else {}
        man.setdefault("stages", {}).setdefault(stage, {}).update(rec)
        if "data_fingerprints" not in man:
            man["data_fingerprints"] = data_fingerprints()
        path.write_text(json.dumps(man, indent=2, default=str))
        fcntl.flock(lk, fcntl.LOCK_UN)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


# ---------------------------------------------------------------- organelle vocabulary
@dataclass(frozen=True)
class OrganelleVocab:
    """The 22 codes in organelles.csv order = the column order of every score matrix."""

    codes: tuple[str, ...]
    go_ids: tuple[str, ...]
    names: tuple[str, ...]

    @classmethod
    def load(cls) -> "OrganelleVocab":
        org = pd.read_csv(RAW / "organelles" / "organelles.csv")
        return cls(tuple(org["code"]), tuple(org["go_id"]), tuple(org["name"]))

    @property
    def go2code(self) -> dict[str, str]:
        return dict(zip(self.go_ids, self.codes))

    @property
    def code2idx(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.codes)}

    def __len__(self) -> int:
        return len(self.codes)


# ---------------------------------------------------------------- score matrices
@dataclass
class ScoreMatrix:
    """Produced by every model, consumed by the harness. NaN row = gene unscorable by this model."""

    model: str
    protocol: str  # "p1" | "p1r" | "p2"
    regime: str  # "strict" | "full" (P1) | "fold_k" (P2)
    seed: int
    genes: list[str]  # ENSG ids of the scored rows
    codes: tuple[str, ...]
    S: np.ndarray  # float32 (n_genes, 22)
    meta: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.model}_{self.protocol}_{self.regime}_seed{self.seed}"

    def save(self, root: Path | None = None) -> Path:
        d = (root or out_dir()) / "scores"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.name}.npz"
        np.savez_compressed(
            path,
            S=self.S.astype(np.float32),
            genes=np.array(self.genes),
            codes=np.array(self.codes),
            info=json.dumps(
                {
                    "model": self.model,
                    "protocol": self.protocol,
                    "regime": self.regime,
                    "seed": self.seed,
                    "meta": self.meta,
                },
                default=_json_default,
            ),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "ScoreMatrix":
        z = np.load(path, allow_pickle=False)
        info = json.loads(str(z["info"]))
        return cls(
            info["model"],
            info["protocol"],
            info["regime"],
            int(info["seed"]),
            [str(g) for g in z["genes"]],
            tuple(str(c) for c in z["codes"]),
            z["S"],
            info.get("meta", {}),
        )


def gene_table() -> pd.DataFrame:
    g = pd.read_csv(IMPORT / "gene_nodes.csv", dtype={"chromosome": str})
    return g.rename(
        columns={
            "ensembl_id:ID(Gene)": "gene",
            "start:long": "start",
            "end:long": "end",
            "length:long": "length",
            "n_interactions:long": "n_interactions",
            ":LABEL": "label",
        }
    )


def located_in_edges(vocab: OrganelleVocab | None = None) -> pd.DataFrame:
    """All 16,085 LOCATED_IN edges as (gene, code, reliability, hpa_term)."""
    vocab = vocab or OrganelleVocab.load()
    l = pd.read_csv(IMPORT / "located_in.csv")
    l = l.rename(columns={":START_ID(Gene)": "gene", ":END_ID(CC)": "go_id"})
    l["code"] = l["go_id"].map(vocab.go2code)
    assert l["code"].notna().all(), "LOCATED_IN target outside the organelle vocabulary"
    return l[["gene", "code", "reliability", "hpa_term"]]


__all__ = [
    "SEED",
    "ROOT",
    "IMPORT",
    "RAW",
    "out_dir",
    "load_config",
    "config_hash",
    "base_parser",
    "resolve_args",
    "seed_everything",
    "select_device",
    "StageLog",
    "sha256_file",
    "update_manifest",
    "write_json",
    "OrganelleVocab",
    "ScoreMatrix",
    "gene_table",
    "located_in_edges",
    "data_fingerprints",
]
