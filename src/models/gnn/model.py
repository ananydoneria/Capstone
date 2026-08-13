"""Temporal propagation-confidence GNN: per-day node encoder + GRU rollup
+ directional edge-pair head.

Forward pass for a window ending at day t: encode each of the trailing
``temporal_window_days`` days' node features through GraphSAGE (or GAT)
message passing over the supply-chain graph (weights shared across days),
then roll the resulting per-day embedding sequence up through a GRU per
node — so the pair score at t reflects recent structural momentum, not just
a single day's snapshot. Score a directed pair (A, B) with an MLP on the
GRU's final [h_A || h_B]. Sigmoid of the logit is the Propagation Confidence
Score: P(shock at A on day t cascades to B within k days).
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, SAGEConv

from src.common.config import Config, load_config


class PropagationGNN(nn.Module):
    def __init__(self, in_dim: int, cfg: Config | None = None):
        super().__init__()
        cfg = cfg or load_config()
        g = cfg.gnn
        conv = {"graphsage": SAGEConv, "gat": GATConv}[g.arch]
        dims = [in_dim] + [g.hidden_dim] * g.num_layers
        self.convs = nn.ModuleList(
            conv(dims[i], dims[i + 1]) for i in range(g.num_layers)
        )
        self.dropout = nn.Dropout(g.dropout)
        self.temporal = nn.GRU(
            input_size=g.hidden_dim, hidden_size=g.temporal_hidden_dim, batch_first=True
        )
        self.edge_head = nn.Sequential(
            nn.Linear(2 * g.temporal_hidden_dim, g.temporal_hidden_dim),
            nn.ReLU(),
            nn.Dropout(g.dropout),
            nn.Linear(g.temporal_hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """One day's node features -> node embeddings. x: [N, F] -> [N, H]."""
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = self.dropout(torch.relu(h))
        return h

    def encode_window(self, x_window: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Trailing window of daily node features -> temporally-rolled-up
        node embeddings. x_window: [W, N, F] (oldest -> newest) -> [N, H_t]."""
        W, N, _ = x_window.shape
        per_day = torch.stack([self.encode(x_window[t], edge_index) for t in range(W)])  # [W, N, H]
        seq = per_day.permute(1, 0, 2)  # [N, W, H]
        _, h_n = self.temporal(seq)     # h_n: [1, N, H_t]
        return h_n.squeeze(0)           # [N, H_t]

    def forward(
        self,
        x_window: torch.Tensor,     # [W, N, F] trailing daily node features, oldest -> newest
        edge_index: torch.Tensor,   # [2, E]
        pairs: torch.Tensor,        # [P, 2] directed (src, dst) node indices
    ) -> torch.Tensor:              # [P] logits
        h = self.encode_window(x_window, edge_index)
        return self.edge_head(torch.cat([h[pairs[:, 0]], h[pairs[:, 1]]], dim=1)).squeeze(-1)
