#!/usr/bin/env python3
"""Launcher of the permuted-attribute control (topology-only perceptron and GraphSAGE), ml_train pattern.

Runs, from the project root and in this order, the two training stages of the control as subprocesses:
  mlp    gnn.train --model mlp  --loss bce --features perm --protocol all   (7 tasks x the configured seeds)
  sage   gnn.train --model sage --loss bce --features perm --protocol all   (same grid)
Each stage prints `PROGRESS {json}` records (a card, one step per finished job, a closing record) that feed the
console (`mlprogress.TrainingConsole`): a header card with the exact parameter count of the built encoder, one live
row per block of jobs, a closing line. The whole module output goes to the log, which ends with the new or changed
artifacts, their SHA-256 and `EXIT <code>`. Ctrl-C is forwarded to the module as SIGINT and closed as `EXIT 130`.

Usage (printed by the agent, run by the author in a terminal that stays open, a herdr pane):
  cd <project root> && .venv/bin/python -u ml/launch_xperm.py [--seeds 20260702 ...] [--stages mlp sage] [--jobs 4]
  --dry-run builds the cards, prints the stage commands and launches nothing.
Resume: relaunch the same command. A stage retrains every job it lists (gnn.train skips nothing), so
`--stages sage` skips a finished mlp stage. The evaluation and comparison stages are run afterwards by the agent
(`make gnn-eval gnn-compare`), because they rewrite the reported aggregates.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "ml", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from mlprogress import LOCK, TrainingConsole, count_parameters, echo  # noqa: E402

NAME = "xperm"
CONFIG = "ml/gnn/config/default.yaml"
ENV = {**os.environ, "PYTHONPATH": "src:ml"}  # the module path convention of the Makefile (GNNPY)
LOG_DIR = "reports/gnn/logs"
ARTIFACTS = [
    "reports/gnn/scores/*_xperm*.npz",
    "reports/gnn/models/*_xperm*",
    "reports/gnn/curves/*_xperm*.csv",
    "reports/gnn/run_manifest.json",
]
EXPECTED = {"mlp": "about 2 min", "sage": "about 20 to 45 min"}  # 4 worker processes x 3 threads, CPU
ECHO = re.compile(r"^(OK |PF-|Traceback|.*Error|.*[Ww]arning)")  # notable module lines shown on the console
PREFIX = "PROGRESS "


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(patterns: list[str]) -> dict[str, float]:
    return {
        path: Path(path).stat().st_mtime
        for pattern in patterns
        for path in glob.glob(str(ROOT / pattern), recursive=True)
        if Path(path).is_file()
    }


def stage_command(kind: str, seeds: list[int] | None, jobs: int) -> list[str]:
    cmd = [sys.executable, "-u", "-m", "gnn.train", "--config", CONFIG, "--protocol", "all"]
    cmd += ["--model", kind, "--loss", "bce", "--features", "perm", "--jobs", str(jobs)]
    if seeds:
        cmd += ["--seeds", *map(str, seeds)]
    return cmd


def model_parameters() -> tuple[dict, dict[str, tuple[int, int]]]:
    """The configuration and the exact parameter counts of the two encoders, built as gnn.train builds them."""
    from gnn.common import load_config
    from gnn.graph import LOC
    from gnn.model import FeatureMLP, HeteroSAGE
    from gnn.train import _context

    cfg = load_config(ROOT / CONFIG)
    index, vocab, _task, X, static_eid = _context("p1", "full")
    n_org, hp_s, hp_m = len(vocab), cfg["sage"], cfg["mlp"]
    sage = HeteroSAGE(
        list(static_eid) + [LOC, ("cc", "rev_located_in", "gene")],
        index.num_nodes(),
        X.shape[1],
        d=hp_s["hidden"],
        L=hp_s["layers"],
        p=hp_s["dropout"],
        n_out=n_org,
    )
    mlp = FeatureMLP(X.shape[1] + n_org, hp_m["hidden"], hp_m.get("dropout", 0.3), n_org)
    return cfg, {"HeteroSAGE": count_parameters(sage), "FeatureMLP": count_parameters(mlp)}


class Monitor:
    """Turns progress records into the console: a card opens a stage, steps feed the live rows, finished closes it."""

    def __init__(self, log=None, parameters: dict[str, tuple[int, int]] | None = None) -> None:
        self.log = log  # the run's log, which also receives the card, the events and the frozen rows
        self.parameters = parameters or {}  # exact counts per encoder, injected into the card
        self.console: TrainingConsole | None = None

    def record(self, r: dict) -> None:
        kind = r.pop("record", None)
        if kind == "card":
            if self.console is not None:
                self.console.close()
            self.console = TrainingConsole(
                total=int(r.pop("total")),
                unit=r.pop("unit", "epoch"),
                best=r.pop("best", None),
                log=self.log,
                standard_metrics=r.pop("standard_metrics", True),
            )
            parameters = r.pop("parameters", None) or self.parameters.get(r.pop("encoder", None))
            self.console.header(
                r.pop("model", "model"),
                parameters=tuple(parameters) if isinstance(parameters, list) else parameters,
                hyperparameters=r.pop("hyperparameters", None),
                chip=r.pop("chip", None),
                **r,
            )
        elif self.console is None:
            return
        elif kind == "step":
            self.console.step(int(r.pop("index")), **r)
        elif kind == "event":
            self.console.event(str(r.get("text", "")))
        elif kind == "finished":
            self.console.close(**r)
            self.console = None

    def line(self, text: str) -> bool:
        """A stdout line of the module; True when it was a progress record."""
        if not text.startswith(PREFIX):
            return False
        try:
            self.record(json.loads(text[len(PREFIX) :]))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return True


def run_stage(name: str, cmd: list[str], log, monitor: Monitor, poll: float) -> int:
    """One stage as a subprocess: its output goes to the log, progress records to the console, notable lines above."""
    started = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log.write(f"\n=== stage {name} started {stamp}: {' '.join(cmd)}\n")
    log.flush()
    echo(f"[{stamp}] stage {name} ({EXPECTED.get(name, '')}): {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    def pump() -> None:
        assert proc.stdout is not None
        for text in proc.stdout:
            if monitor.line(text.rstrip("\n")):  # a progress record: rendered by the console, which logs the rows
                continue
            with LOCK:  # the console writes to the same log from the main thread
                log.write(text)
                log.flush()
            if ECHO.match(text):
                echo(text.rstrip())

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        while proc.poll() is None:
            time.sleep(poll)
    except KeyboardInterrupt:
        echo("interrupted: stopping the module; relaunch the same command to run the stage again")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
        reader.join(timeout=30)
        log.write(f"=== stage {name} interrupted\n")
        return 130
    reader.join(timeout=30)
    seconds = time.time() - started
    line = f"=== stage {name} EXIT {proc.returncode} after {seconds / 3600:.2f} h ({seconds:.0f} s)"
    log.write(line + "\n")
    log.flush()
    echo(line)
    return proc.returncode


def launch(stages: list[tuple[str, list[str]]], plan: str, parameters: dict, poll: float) -> int:
    before = snapshot(ARTIFACTS)
    log_dir = ROOT / LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{NAME}_{datetime.now().strftime('%y%m%d_%H%M')}.log"
    code = 0
    with log_path.open("w", encoding="utf-8") as log:
        monitor = Monitor(log, parameters)
        head = (
            f"{NAME} launcher: {plan}; {len(before)} xperm artifacts present before (gnn.train retrains and "
            f"rewrites them); log {log_path.relative_to(ROOT)}"
        )
        log.write(head + "\n")
        print(head)
        for name, cmd in stages:
            code = run_stage(name, cmd, log, monitor, poll)
            if code != 0:
                break
        if monitor.console is not None:  # a stage that ended without its finished record keeps its partial row
            monitor.console.close()
        after = snapshot(ARTIFACTS)
        changed = sorted(p for p, mtime in after.items() if before.get(p) != mtime)
        log.write(f"\n=== artifacts: {len(before)} -> {len(after)}; new or changed: {len(changed)}\n")
        for path in changed:
            log.write(f"  {Path(path).relative_to(ROOT)}  {sha256(Path(path))}\n")
        log.write(f"EXIT {code}\n")
    print(f"artifacts new or changed: {len(changed)}")
    print(f"log: {log_path.relative_to(ROOT)}")
    print(f"EXIT {code}")
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launcher of the permuted-attribute control (ml_train pattern); run from a terminal."
    )
    parser.add_argument("--stages", nargs="+", default=["mlp", "sage"], choices=["mlp", "sage"])
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="run seeds (default: the configured 5)")
    parser.add_argument("--jobs", type=int, default=4, help="parallel worker processes of gnn.train")
    parser.add_argument("--poll", type=float, default=2.0, help="seconds between process polls")
    parser.add_argument("--dry-run", action="store_true", help="build the cards and print the commands only")
    args = parser.parse_args(argv)
    cfg, parameters = model_parameters()  # refuses early (traceback) when the graph, features or splits are missing
    seeds = args.seeds or list(cfg["sage"]["seeds"])
    n_tasks = len(cfg["graph"]["regimes"]) + int(cfg["p2"]["n_folds"])
    stages = [(kind, stage_command(kind, args.seeds, args.jobs)) for kind in args.stages]
    plan = (
        f"stages {args.stages}; seeds {seeds}; {n_tasks * len(seeds)} jobs per stage ({n_tasks} tasks x "
        f"{len(seeds)} seeds); expected " + ", ".join(f"{k} {EXPECTED[k]}" for k in args.stages)
        + f"; device {cfg.get('device')} with {args.jobs} processes x {cfg.get('threads')} threads"
    )
    if args.dry_run:
        from argparse import Namespace

        from gnn.train import _card, build_jobs

        print(f"dry run: {plan}")
        for enc, (total, trainable) in parameters.items():
            print(f"{enc}: {total:,} parameters ({trainable:,} trainable)")
        monitor = Monitor(None, parameters)
        for kind in args.stages:
            jobs = build_jobs(cfg, kind, "bce", "all", None, args.seeds, None, "perm")
            card = _card(cfg, Namespace(model=kind, loss="bce", features="perm", jobs=args.jobs), jobs)
            monitor.record({"record": "card", **card})
            monitor.record({"record": "finished", "dry_run": True})
        for name, cmd in stages:
            print(f"stage {name}: {' '.join(cmd)}")
        print("dry run: nothing launched")
        return 0
    return launch(stages, plan, parameters, args.poll)


if __name__ == "__main__":
    sys.exit(main())
