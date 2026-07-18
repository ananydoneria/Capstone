"""Propagation-confidence GNN: node encoder + directional edge-pair head.

Forward pass on day t: encode all node feature vectors through GraphSAGE (or
GAT) message passing over the supply-chain graph, then score a directed pair
(A, B) with an MLP on [h_A || h_B]. Sigmoid of the logit is the Propagation
Confidence Score: P(shock at A on day t cascades to B within k days).
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
        self.edge_head = nn.Sequential(
            nn.Linear(2 * g.hidden_dim, g.hidden_dim),
            nn.ReLU(),
            nn.Dropout(g.dropout),
            nn.Linear(g.hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = self.dropout(torch.relu(h))
        return h

    def forward(
        self,
        x: torch.Tensor,            # [N, F] node features for one day
        edge_index: torch.Tensor,   # [2, E]
        pairs: torch.Tensor,        # [P, 2] directed (src, dst) node indices
    ) -> torch.Tensor:              # [P] logits
        h = self.encode(x, edge_index)
        return self.edge_head(torch.cat([h[pairs[:, 0]], h[pairs[:, 1]]], dim=1)).squeeze(-1)
