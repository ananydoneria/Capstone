"""Temporal propagation-confidence GNN: per-day node encoder + GRU rollup
+ directional edge-pair head.

Forward pass for a window ending at day t: encode each of the trailing
``temporal_window_days`` days' node features through GraphSAGE / GraphConv /
GAT message passing over the supply-chain graph (weights shared across days),
then roll the resulting per-day embedding sequence up through a GRU per
node — so the pair score at t reflects recent structural momentum, not just
a single day's snapshot. Score a directed pair (A, B) with an MLP on
[h_A || h_B || pair features || market context]. Sigmoid of the logit is the
Propagation Confidence Score: P(shock at A on day t cascades to B within k days).

``graphconv`` is the only arch that consumes the curated edge weights
(relationships.csv ``weight``): neighbour messages are scaled by link strength
before mean aggregation. Pair features (rolling correlation, link weight,
direction) and the market-context vector are optional and only change the
head's input width; with both absent the model is exactly the original.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, GraphConv, SAGEConv

from src.common.config import Config, load_config


class PropagationGNN(nn.Module):
    def __init__(self, in_dim: int, cfg: Config | None = None, pair_dim: int = 0, ctx_dim: int = 0):
        super().__init__()
        cfg = cfg or load_config()
        g = cfg.gnn
        dims = [in_dim] + [g.hidden_dim] * g.num_layers
        if g.arch == "graphconv":
            self.convs = nn.ModuleList(
                GraphConv(dims[i], dims[i + 1], aggr="mean") for i in range(g.num_layers)
            )
        else:
            conv = {"graphsage": SAGEConv, "gat": GATConv}[g.arch]
            self.convs = nn.ModuleList(conv(dims[i], dims[i + 1]) for i in range(g.num_layers))
        self.uses_edge_weight = g.arch == "graphconv"
        self.pair_dim, self.ctx_dim = pair_dim, ctx_dim
        self.dropout = nn.Dropout(g.dropout)
        self.temporal = nn.GRU(
            input_size=g.hidden_dim, hidden_size=g.temporal_hidden_dim, batch_first=True
        )
        self.edge_head = nn.Sequential(
            nn.Linear(2 * g.temporal_hidden_dim + pair_dim + ctx_dim, g.temporal_hidden_dim),
            nn.ReLU(),
            nn.Dropout(g.dropout),
            nn.Linear(g.temporal_hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor,
               edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        """One day's node features -> node embeddings. x: [N, F] -> [N, H]."""
        h = x
        for i, conv in enumerate(self.convs):
            if self.uses_edge_weight and edge_weight is not None:
                h = conv(h, edge_index, edge_weight)
            else:
                h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = self.dropout(torch.relu(h))
        return h

    def encode_window(self, x_window: torch.Tensor, edge_index: torch.Tensor,
                      edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        """Trailing window of daily node features -> temporally-rolled-up
        node embeddings. x_window: [W, N, F] (oldest -> newest) -> [N, H_t]."""
        W = x_window.shape[0]
        per_day = torch.stack([self.encode(x_window[t], edge_index, edge_weight) for t in range(W)])
        seq = per_day.permute(1, 0, 2)  # [N, W, H]
        _, h_n = self.temporal(seq)     # h_n: [1, N, H_t]
        return h_n.squeeze(0)           # [N, H_t]

    def forward(
        self,
        x_window: torch.Tensor,                   # [W, N, F] trailing daily node features, oldest -> newest
        edge_index: torch.Tensor,                 # [2, E]
        pairs: torch.Tensor,                      # [P, 2] directed (src, dst) node indices
        edge_weight: torch.Tensor | None = None,  # [E], used by graphconv only
        pair_feat: torch.Tensor | None = None,    # [P, pair_dim]
        ctx: torch.Tensor | None = None,          # [ctx_dim] market context at day t
    ) -> torch.Tensor:                            # [P] logits
        h = self.encode_window(x_window, edge_index, edge_weight)
        parts = [h[pairs[:, 0]], h[pairs[:, 1]]]
        if self.pair_dim:
            parts.append(pair_feat)
        if self.ctx_dim:
            parts.append(ctx.unsqueeze(0).expand(len(pairs), -1))
        return self.edge_head(torch.cat(parts, dim=1)).squeeze(-1)
