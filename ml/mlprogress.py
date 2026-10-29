#!/usr/bin/env python3
"""Sober block-wise console for training runs (author's style rule of 2026-10-17, live block rows of 2026-10-17;
standard library only).

A run prints three things and nothing else: a header card with the model and its main characteristics (for a
neural network always the parameter count; the standard hyperparameters on their own line, apart from the
algorithm-specific protocol; the device with its GPU or chip model beside it), one row per block of steps (a block is 10 to 20 % of the total, so
that a run of any size prints 5 to 10 rows: epochs for a deep model; rounds, folds, trials, models of a family or
a group of shards for other algorithms), and sparse event lines (early stop, a recovery, a collapse, a refusal).
The row of the block in progress is live on a terminal: it is redrawn in place after every step, its ASCII bar
fills from empty to full over the steps of that block, the counter shows the current step over the total, the
percentage is the overall completion, and the learning metrics are those of the last step with the best value
seen so far. When the block closes the row is frozen and the next block opens a new row below it. Off a terminal
(a pipe, `tee`, a log file) only the frozen rows are written, so a log never carries redraws. A closing line gives
the best step, the best metric and the total time. The block size, the unit and the metrics are chosen per
algorithm; the shape stays the same so that an advanced reader follows any run at a glance without a flooded
terminal.

Prototype use:
    console = TrainingConsole(total=60, fraction=0.1, unit="epoch", best={"val_auroc": "max", "val_loss": "min"})
    console.header("M2 convolutional + bidirectional LSTM", parameters=count_parameters(model),
                   hyperparameters={"optimizer": "Adam", "learning_rate": 0.0005, "batch": 256, "max_epochs": 60},
                   task="expr_generic from the 10.5 kb promoter", train="13,964 genes", val="2,084", test="2,050",
                   protocol="early stop on val AUROC, patience 5, warm-up 15", device="cuda")   # cuda (<GPU name>)
    for epoch in range(1, 61):
        ...
        console.step(epoch, train_loss=loss, val_loss=vloss, val_auroc=auroc)   # redraws the live row of the block
    console.event("early stop at epoch 51; best epoch 46 restored")
    console.close()
Other output printed while a row is live goes through `echo(text)`, which keeps the row below it. `log=` gives the run's
log file, which then receives the card, the events, the frozen rows and the closing line (never a redraw).
The closing line carries accuracy, precision, recall and F1 of the final model: `close(**classification_metrics(y, yhat),
...)`; `standard_metrics=False` drops them for runs they do not describe (regression, ranking, inference counts).
Demo: python3 development/scripts/mlprogress.py --demo [--delay 0.2]
"""
from __future__ import annotations

import argparse
import math
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, TextIO

_LOCK = threading.RLock()
_ACTIVE: "TrainingConsole | None" = None  # the console whose block row is live on the terminal
LOCK = _LOCK  # for launchers that write the same log file from another thread (the module-output pump)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 10 else f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _scaled(n: int) -> str:
    for size, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if n >= size:
            return f"{n / size:.2f} {suffix}"
    return str(n)


def count_parameters(model: Any) -> tuple[int, int]:
    """Total and trainable parameters of a neural network (any object with `parameters()` yielding tensors)."""
    params = list(model.parameters())
    return (
        int(sum(p.numel() for p in params)),
        int(sum(p.numel() for p in params if getattr(p, "requires_grad", True))),
    )


def parameters_text(parameters: int | tuple[int, int]) -> str:
    total, trainable = parameters if isinstance(parameters, tuple) else (parameters, parameters)

    def one(n: int) -> str:
        return f"{n:,} ({_scaled(n)})" if n >= 1000 else f"{n:,}"

    return one(total) if trainable == total else f"{one(total)}; trainable {one(trainable)}"


def chip_name(device: str) -> str | None:
    """GPU or processor model behind a device string: the CUDA device name, the Apple chip, or the CPU model."""
    kind = device.split(":")[0]
    if kind == "cuda":
        try:
            import torch

            return torch.cuda.get_device_name(int(device.split(":")[1]) if ":" in device else 0)
        except Exception:
            return None
    if platform.system() == "Darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
            ).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def device_text(device: str, chip: str | None = None) -> str:
    """The device with its GPU or chip model beside it, e.g. `cuda (NVIDIA GeForce RTX 2080 Ti)`."""
    name = chip or chip_name(device)
    return f"{device} ({name})" if name else device


def hyperparameters_text(hyperparameters: dict[str, Any] | str) -> str:
    if isinstance(hyperparameters, str):
        return hyperparameters
    return ", ".join(
        f"{key.replace('_', ' ')} {value:g}" if isinstance(value, float) else f"{key.replace('_', ' ')} {value}"
        for key, value in hyperparameters.items()
    )


def classification_metrics(
    y_true: Any, y_pred: Any, positive: Any = 1, average: str = "binary"
) -> dict[str, float]:
    """Accuracy, precision, recall and F1 from true and predicted labels (predictions already thresholded at the
    protocol's decision threshold). `average` is "binary" (the `positive` class) or "macro" (unweighted mean over the
    classes present); a ratio with an empty denominator is 0.0, as in scikit-learn's zero_division=0."""
    pairs = list(zip(list(y_true), list(y_pred)))
    if not pairs:
        raise ValueError("classification_metrics: no predictions")

    def one(label: Any) -> tuple[float, float, float]:
        tp = sum(1 for t, q in pairs if q == label and t == label)
        fp = sum(1 for t, q in pairs if q == label and t != label)
        fn = sum(1 for t, q in pairs if q != label and t == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    if average == "binary":
        precision, recall, f1 = one(positive)
    elif average == "macro":
        scores = [one(label) for label in sorted({t for t, _ in pairs} | {q for _, q in pairs}, key=str)]
        precision, recall, f1 = (sum(column) / len(scores) for column in zip(*scores))
    else:
        raise ValueError('average: "binary" or "macro"')
    accuracy = sum(1 for t, q in pairs if t == q) / len(pairs)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def echo(text: str, fallback: Callable[[str], None] | None = None) -> None:
    """Print a line from outside the console (a stage line, a module message) without breaking a live row."""
    with _LOCK:
        if _ACTIVE is not None and _ACTIVE.live:
            _ACTIVE._interject(text)
        elif fallback is not None:
            fallback(text)
        else:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()


class TrainingConsole:
    """Header card, one live row per block, sparse events, a closing line."""

    def __init__(
        self,
        total: int,
        block: int | None = None,
        fraction: float = 0.1,
        unit: str = "epoch",
        best: dict[str, str] | None = None,
        width: int = 24,
        stream: TextIO | None = None,
        live: bool | None = None,
        log: TextIO | None = None,
        standard_metrics: bool = True,
    ) -> None:
        self.total, self.unit, self.width = max(1, total), unit, width
        if not 0.05 <= fraction <= 0.25:
            raise ValueError("fraction: a block is 10 to 20 % of the total (0.05 to 0.25 admitted)")
        self.fraction = fraction
        if block is not None:  # an explicit block size; the last block takes the remainder
            self.block = max(1, block)
            self.ends = list(range(self.block, self.total, self.block)) + [self.total]
        else:  # round(1 / fraction) blocks of near-equal size, so that the run prints that many rows
            n = min(self.total, max(1, round(1 / fraction)))
            self.ends = sorted({round(self.total * i / n) for i in range(1, n + 1)})
            self.block = self.ends[0]
        self.best_rule = best or {}
        self.best: dict[str, tuple[float, int]] = {}
        self.stream = stream or sys.stdout
        self.live = self.stream.isatty() if live is None else live
        self.log = log  # the run's log: card, events, frozen rows and closing line, never a redraw
        # the closing line carries accuracy, precision, recall and F1 of the final model (n/a when not supplied);
        # False for runs those metrics do not describe (regression, ranking and embeddings, inference counts)
        self.standard_metrics = standard_metrics
        self.started = time.monotonic()
        self.done = 0
        self.rows = 0
        self.current: int | None = None  # index of the block whose row is open
        self.pending: str | None = None  # text of the open row
        self.drawn = 0  # terminal lines the open row occupies

    # ---- header card: the model and its main characteristics, as the caller states them; `hyperparameters` holds
    # the standard training values of the model (optimizer, learning rate, batch, epochs, loss, regularization),
    # `protocol` (a card key) the algorithm-specific policy (stopping rule, warm-up, guards, budget); `device` is
    # printed with its GPU or chip model beside it (`chip`, or detected)
    def header(
        self,
        model: str,
        parameters: int | tuple[int, int] | None = None,
        hyperparameters: dict[str, Any] | str | None = None,
        chip: str | None = None,
        **card: Any,
    ) -> None:
        self._print(f"model: {model}")
        if parameters is not None:
            self._print(f"  parameters: {parameters_text(parameters)}")
        if hyperparameters:
            self._print(f"  hyperparameters: {hyperparameters_text(hyperparameters)}")
        for key, value in card.items():
            if key == "device" and isinstance(value, str):
                value = device_text(value, chip)
            self._print(f"  {key.replace('_', ' ')}: {str(value) if isinstance(value, int) else _fmt(value)}")
        sizes = sorted({b - a for a, b in zip([0] + self.ends, self.ends)})
        size = f"{sizes[0]}" if len(sizes) == 1 else f"{sizes[0]} to {sizes[-1]}"
        share = 100 * self.ends[0] / self.total
        self._print(
            f"  run: {self.total:,} {self.unit}s in {len(self.ends)} blocks of {size} {self.unit}s "
            f"(about {share:.0f} % of the run each), one row per block"
        )
        self._print("-" * 100)

    # ---- one step with its metrics; the live row of its block is redrawn, a closed block freezes its row
    def step(self, index: int, **metrics: Any) -> None:
        with _LOCK:
            self.done = index
            for key, value in metrics.items():
                rule = self.best_rule.get(key)
                if rule and isinstance(value, (int, float)):
                    current = self.best.get(key)
                    better = current is None or (
                        value > current[0] if rule == "max" else value < current[0]
                    )
                    if better:
                        self.best[key] = (float(value), index)
            position = next((i for i, end in enumerate(self.ends) if index <= end), len(self.ends) - 1)
            if self.current is not None and position != self.current:
                self._freeze()
            self.current = position
            self.pending = self.row(index, **metrics)
            if index >= self.ends[position]:
                self._freeze()
            elif self.live:
                self._draw()

    def row(self, index: int, **metrics: Any) -> str:
        position = self.current if self.current is not None else 0
        first = (self.ends[position - 1] if position else 0) + 1
        last = self.ends[position]
        filled = int(round(self.width * (min(index, last) - first + 1) / (last - first + 1)))
        bar = "=" * filled + (">" if filled < self.width else "") + " " * max(0, self.width - filled - 1)
        per_unit = (time.monotonic() - self.started) / max(index, 1)
        eta = per_unit * max(self.total - index, 0)
        cells = [
            f"block {position + 1}/{len(self.ends)} [{bar}] {self.unit} {index:,}/{self.total:,}",
            f"{100 * min(1.0, index / self.total):5.1f} %",
        ]
        for key, value in metrics.items():
            text = f"{key}={_fmt(value)}"
            if key in self.best:
                best_value, at = self.best[key]
                text += f" (best {_fmt(best_value)} @{at})"
            cells.append(text)
        cells.append(f"{per_unit:.1f} s/{self.unit}")
        cells.append(f"ETA {_hms(eta)}")
        return " | ".join(cells)

    # ---- sparse events: early stop, recovery, collapse, refusal; never per step
    def event(self, text: str) -> None:
        with _LOCK:
            if self.live and self.pending:
                self._interject(f"  ! {text}")
                self._log(f"  ! {text}")
            else:
                self._print(f"  ! {text}")

    def close(
        self,
        accuracy: float | None = None,
        precision: float | None = None,
        recall: float | None = None,
        f1: float | None = None,
        split: str = "test",
        average: str | None = None,
        **summary: Any,
    ) -> None:
        """The closing line: steps and time, the four standard metrics of the final model on `split` (`average` names
        the averaging of a multiclass task), the best value of each tracked metric, then the caller's summary."""
        global _ACTIVE
        with _LOCK:
            self._freeze()  # an early stop leaves the last block partial: its row is kept as it stands
            if _ACTIVE is self:
                _ACTIVE = None
            elapsed = time.monotonic() - self.started
            cells = [f"done: {self.done:,}/{self.total:,} {self.unit}s in {_hms(elapsed)}"]
            if self.standard_metrics:
                values = (("accuracy", accuracy), ("precision", precision), ("recall", recall), ("F1", f1))
                label = f"{split} {average}" if average else split
                cells.append(
                    f"{label}: "
                    + ", ".join(f"{name} {'n/a' if v is None else f'{float(v):.4f}'}" for name, v in values)
                )
            for key, (value, at) in self.best.items():
                cells.append(f"best {key} {_fmt(value)} @{at}")
            for key, value in summary.items():
                cells.append(f"{key}={_fmt(value)}")
            self._print("-" * 100)
            self._print(" | ".join(cells))

    # ---- terminal mechanics: the open row is erased and rewritten in place, wrapped lines included
    def _erase(self) -> None:
        if self.drawn:
            up = f"\x1b[{self.drawn - 1}A" if self.drawn > 1 else ""
            self.stream.write(f"\r{up}\x1b[J")
            self.drawn = 0

    def _draw(self) -> None:
        global _ACTIVE
        if not self.pending:
            return
        self._erase()
        columns = max(20, shutil.get_terminal_size((200, 24)).columns)
        self.stream.write(self.pending)
        self.stream.flush()
        self.drawn = max(1, math.ceil(len(self.pending) / columns))
        _ACTIVE = self

    def _interject(self, text: str) -> None:
        self._erase()
        self.stream.write(text + "\n")
        self._draw()
        self.stream.flush()

    def _freeze(self) -> None:
        if self.pending is None:
            return
        self._erase()
        self.stream.write(self.pending + "\n")
        self.stream.flush()
        self._log(self.pending)
        self.pending, self.current = None, None
        self.rows += 1

    def _print(self, text: str) -> None:
        with _LOCK:
            self.stream.write(text + "\n")
            self.stream.flush()
            self._log(text)

    def _log(self, text: str) -> None:
        if self.log is not None and self.log is not self.stream:
            self.log.write(text + "\n")
            self.log.flush()


def demo(delay: float = 0.01) -> None:
    """A simulated 60-epoch deep-model run, ten live block rows of 6 epochs, an early stop at 51 inside block 9."""
    import random

    random.seed(1)
    console = TrainingConsole(
        total=60, fraction=0.1, unit="epoch", best={"val_auroc": "max", "val_loss": "min"}
    )
    console.header(
        "M2 convolutional + bidirectional LSTM",
        parameters=498_561,
        hyperparameters={
            "optimizer": "Adam",
            "learning_rate": 0.0005,
            "batch": 256,
            "max_epochs": 60,
            "loss": "binary cross-entropy with logits",
            "gradient_clip": 1.0,
            "dropout": 0.2,
            "seed": 20260702,
        },
        task="expr_generic from the 10.5 kb promoter window (one-hot)",
        split="train 13,964 / val 2,084 / test 2,050 genes (chromosome held-out)",
        protocol="early stop on val AUROC, patience 5, warm-up 15, best epoch restored; collapse guard",
        device="cuda",
        chip="NVIDIA GeForce RTX 2080 Ti",
        deterministic=True,
    )
    for epoch in range(1, 61):
        time.sleep(delay)
        loss = 0.70 * math.exp(-epoch / 18) + 0.28 + random.uniform(-0.01, 0.01)
        vloss = 0.72 * math.exp(-epoch / 22) + 0.33 + random.uniform(-0.015, 0.015)
        auroc = 0.86 - 0.36 * math.exp(-epoch / 12) + random.uniform(-0.01, 0.01)
        console.step(epoch, train_loss=loss, val_loss=vloss, val_auroc=auroc)
        if epoch == 51:
            console.event("early stop at epoch 51 (patience 5); weights of the best epoch restored")
            break
    labels = [random.random() < 0.55 for _ in range(2050)]  # a simulated test split, thresholded predictions
    predicted = [y if random.random() < 0.8 else not y for y in labels]
    console.close(
        **classification_metrics(labels, predicted, positive=True),
        test_auroc=0.8559,
        checkpoint="models/clf_M2_20260702_cuda_<date>.pt",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Block-wise training console with live block rows (prototype); --demo prints a simulated run."
    )
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--delay", type=float, default=0.01, help="demo: seconds per simulated epoch")
    args = parser.parse_args(argv)
    if args.demo:
        demo(args.delay)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
