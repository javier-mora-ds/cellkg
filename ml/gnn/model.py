"""Heterogeneous GraphSAGE encoder with a 22-logit multi-label head, and the feature-only perceptron."""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import HeteroConv, SAGEConv

EMBEDDED_TYPES = ("cc", "bp", "mf", "chromosome")


class HeteroSAGE(nn.Module):
    """Inductive encoder: genes enter through their attribute vector (no per-gene embedding)."""

    def __init__(self, edge_types: list[tuple[str, str, str]], num_nodes: dict[str, int], d_gene_in: int,
                 d: int = 128, L: int = 2, p: float = 0.3, n_out: int = 22):
        super().__init__()
        self.proj = nn.Linear(d_gene_in, d)
        self.emb = nn.ModuleDict({nt: nn.Embedding(num_nodes[nt], d) for nt in EMBEDDED_TYPES})
        node_types = ("gene",) + EMBEDDED_TYPES
        self.convs = nn.ModuleList(
            HeteroConv({et: SAGEConv(d, d, aggr="mean", root_weight=True) for et in edge_types}, aggr="sum")
            for _ in range(L))
        self.norms = nn.ModuleList(nn.ModuleDict({nt: nn.LayerNorm(d) for nt in node_types}) for _ in range(L))
        self.drop = nn.Dropout(p)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, n_out))

    def encode(self, x_gene: Tensor, edge_index_dict: dict) -> dict[str, Tensor]:
        h = {"gene": self.proj(x_gene)}
        h.update({nt: self.emb[nt].weight for nt in EMBEDDED_TYPES})
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, edge_index_dict)
            h = {nt: h[nt] + self.drop(F.relu(norm[nt](out[nt]))) if nt in out else h[nt] for nt in h}
        return h

    def forward(self, x_gene: Tensor, edge_index_dict: dict) -> Tensor:
        return self.head(self.encode(x_gene, edge_index_dict)["gene"])   # logits Z [N_gene, 22]


class FeatureMLP(nn.Module):
    """x_gene (425) || k_g (22) -> 256 -> 256 -> 22: the same information minus message passing."""

    def __init__(self, d_in: int, hidden: int = 256, p: float = 0.3, n_out: int = 22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, n_out))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
