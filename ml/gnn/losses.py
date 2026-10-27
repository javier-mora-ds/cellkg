"""Loss family (plan §5.E): per-pair terms on logits, averaged over unmasked pairs.

Class statistics come from the training split of the current protocol / fold only.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def masked_mean(l: Tensor, M: Tensor) -> Tensor:
    return (l * M).sum() / M.sum().clamp(min=1.0)


def bce(z: Tensor, y: Tensor, M: Tensor) -> Tensor:
    return masked_mean(F.binary_cross_entropy_with_logits(z, y, reduction="none"), M)


def logit_adjusted_bce(z: Tensor, y: Tensor, M: Tensor, lpo: Tensor, tau: float = 1.0) -> Tensor:
    """Menon et al. 2021: the prior is added during training only; inference uses the raw z."""
    return bce(z + tau * lpo, y, M)


def weighted_bce(z: Tensor, y: Tensor, M: Tensor, pos_w: Tensor) -> Tensor:
    return masked_mean(F.binary_cross_entropy_with_logits(z, y, pos_weight=pos_w, reduction="none"), M)


def class_balanced_bce(z: Tensor, y: Tensor, M: Tensor, r: Tensor) -> Tensor:
    return masked_mean(F.binary_cross_entropy_with_logits(z, y, reduction="none") * r, M)


def focal(z: Tensor, y: Tensor, M: Tensor, gamma: float = 2.0, alpha: float = 0.25) -> Tensor:
    p = torch.sigmoid(z)
    ce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    pt = y * p + (1 - y) * (1 - p)
    at = y * alpha + (1 - y) * (1 - alpha)
    return masked_mean(at * (1 - pt) ** gamma * ce, M)


def asl(z: Tensor, y: Tensor, M: Tensor, gamma_pos: float = 0.0, gamma_neg: float = 4.0,
        margin: float = 0.05, eps: float = 1e-8) -> Tensor:
    p = torch.sigmoid(z)
    pm = (p - margin).clamp(min=0)
    lp = y * (1 - p) ** gamma_pos * torch.log(p.clamp(min=eps))
    ln = (1 - y) * pm ** gamma_neg * torch.log((1 - pm).clamp(min=eps))
    return -masked_mean(lp + ln, M)


class ClassStats:
    """n_o, N, the Laplace-smoothed prevalence pi_o = (n_o + 1)/(N + 2) and its log prior odds."""

    def __init__(self, Y_train_rows: np.ndarray, supported: np.ndarray):
        self.n = Y_train_rows.sum(0).astype(np.float64)
        self.N = float(Y_train_rows.shape[0])
        self.pi = (self.n + 1) / (self.N + 2)
        self.lpo = np.log(self.pi / (1 - self.pi))
        self.supported = supported.astype(bool)


def make_loss(name: str, stats: ClassStats, cfg: dict):
    """Return loss_fn(z, y, M) for a named member of the family."""
    lc = cfg["losses"]
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    if name == "bce":
        return bce
    if name == "logit_adjusted":
        lpo, tau = t(stats.lpo), lc["logit_adjusted"]["tau"]
        return lambda z, y, M: logit_adjusted_bce(z, y, M, lpo, tau)
    if name == "weighted":
        w = ((stats.N - stats.n) / np.maximum(stats.n, 1)) ** lc["weighted"]["tau"]
        pos_w = t(np.minimum(w, lc["weighted"]["max_pos_weight"]))
        return lambda z, y, M: weighted_bce(z, y, M, pos_w)
    if name == "class_balanced":
        beta = lc["class_balanced"]["beta"]
        r = (1 - beta) / (1 - beta ** np.maximum(stats.n, 1))
        r = t(r / r[stats.supported].mean())
        return lambda z, y, M: class_balanced_bce(z, y, M, r)
    if name == "focal":
        return lambda z, y, M: focal(z, y, M, **lc["focal"])
    if name == "asl":
        return lambda z, y, M: asl(z, y, M, **lc["asl"])
    raise ValueError(f"unknown loss {name}")
