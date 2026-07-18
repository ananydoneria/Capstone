"""Bridge from pipeline artifacts (features.parquet + relationships.csv)
to PyTorch tensors for the propagation GNN.

Everything is tiny (15 nodes, ~740 days), so the whole dataset lives in RAM
as dense tensors; per-day node-feature matrices are z-normalized using
TRAIN-period statistics only.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from src.common.config import Config, load_config
from src.data_pipeline.features import load_features
from src.data_pipeline.relationships import build_graph, to_edge_index
from src.models.gnn.labels import CascadeSample, build_samples, shock_matrix, temporal_split


@dataclass
class GraphData:
    tickers: list[str]
    dates: list[pd.Timestamp]
    x: torch.Tensor            # [T, N, F] z-normalized node features
    edge_index: torch.Tensor   # [2, E] symmetrized message-passing edges
    edge_weight: torch.Tensor  # [E]
    directed_pairs: list[tuple[int, int]]   # supervision edges as node indices
    train: list[CascadeSample]
    val: list[CascadeSample]
    norm_mean: torch.Tensor    # [F] train-period stats (persisted with weights)
    norm_std: torch.Tensor     # [F]

    def day_index(self, date: pd.Timestamp) -> int:
        return self._date_pos[date]

    def __post_init__(self) -> None:
        self._date_pos = {d: i for i, d in enumerate(self.dates)}


def features_tensor(
    features: pd.DataFrame, tickers: list[str]
) -> tuple[list[pd.Timestamp], torch.Tensor]:
    """Long-format features -> (dates, [T, N, F] tensor), ticker order enforced."""
    dates = list(features.index.get_level_values("date").unique())
    wide = features.unstack("ticker")  # columns: (feature, ticker)
    blocks = [
        torch.tensor(wide[name][tickers].to_numpy(), dtype=torch.float32)
        for name in features.columns
    ]
    return dates, torch.stack(blocks, dim=-1)


def supervision_pairs(cfg: Config) -> list[tuple[str, str]]:
    """Directed (src, dst) ticker pairs the edge head scores, config-aware."""
    from src.data_pipeline.relationships import build_graph

    g = build_graph(cfg)
    pairs = [(u, v) for u, v in g.edges()]
    if cfg.gnn.bidirectional_supervision:
        pairs += [(v, u) for u, v in g.edges()]
    return pairs


def build_dataset(cfg: Config | None = None) -> GraphData:
    cfg = cfg or load_config()
    features = load_features(cfg)
    tickers = cfg.universe.tickers

    g = build_graph(cfg)
    ei, ew = to_edge_index(g, tickers, symmetrize=True)
    pos = {t: i for i, t in enumerate(tickers)}
    directed_edges = supervision_pairs(cfg)
    directed_pairs = [(pos[u], pos[v]) for u, v in directed_edges]

    shocks = shock_matrix(features, cfg)
    samples = build_samples(shocks, directed_edges, cfg)
    train, val = temporal_split(samples, cfg)

    dates, x = features_tensor(features, tickers)

    # Normalize with train-period statistics only (no val leakage)
    train_dates = {s.date for s in train}
    train_mask = torch.tensor([d in train_dates for d in dates])
    flat = x[train_mask].reshape(-1, x.shape[-1])
    mean, std = flat.mean(dim=0), flat.std(dim=0).clamp_min(1e-8)
    x = (x - mean) / std

    return GraphData(
        tickers=tickers,
        dates=dates,
        x=x,
        edge_index=torch.tensor(ei, dtype=torch.long),
        edge_weight=torch.tensor(ew, dtype=torch.float32),
        directed_pairs=directed_pairs,
        train=train,
        val=val,
        norm_mean=mean,
        norm_std=std,
    )
