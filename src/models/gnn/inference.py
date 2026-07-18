"""Phase 2.5: full-timeline GNN inference cache.

Loads the trained weights + persisted train-period normalization stats, runs
one forward pass per day over the whole feature timeline, and caches sigmoid
Propagation Confidence Scores per directed pair to
``data/processed/gnn_scores.parquet`` (index=date, columns "SRC->DST").

After this, the GNN is never invoked downstream (compute guardrail).

Run:  python scripts/cache_gnn_scores.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.common.config import Config, load_config
from src.common.seeding import set_global_seed
from src.data_pipeline.features import load_features
from src.data_pipeline.relationships import build_graph, to_edge_index
from src.models.gnn.graph_dataset import features_tensor, supervision_pairs
from src.models.gnn.model import PropagationGNN
from src.models.gnn.train import WEIGHTS_DIR


def scores_path(cfg: Config) -> Path:
    return Path(cfg.data.processed_dir) / "gnn_scores.parquet"


def run_inference(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    set_global_seed(cfg.project.seed)
    tickers = cfg.universe.tickers

    features = load_features(cfg)
    dates, x = features_tensor(features, tickers)
    norm = torch.load(WEIGHTS_DIR / "feature_norm.pt", weights_only=True)
    x = (x - norm["mean"]) / norm["std"]

    g = build_graph(cfg)
    ei, _ = to_edge_index(g, tickers, symmetrize=True)
    edge_index = torch.tensor(ei, dtype=torch.long)
    pair_names = supervision_pairs(cfg)
    pos = {t: i for i, t in enumerate(tickers)}
    pairs = torch.tensor([[pos[u], pos[v]] for u, v in pair_names], dtype=torch.long)

    model = PropagationGNN(in_dim=x.shape[-1], cfg=cfg)
    model.load_state_dict(torch.load(WEIGHTS_DIR / "propagation_gnn.pt", weights_only=True))
    model.eval()

    with torch.no_grad():
        scores = torch.stack(
            [torch.sigmoid(model(x[t], edge_index, pairs)) for t in range(len(dates))]
        )

    out = pd.DataFrame(
        scores.numpy(),
        index=pd.DatetimeIndex(dates, name="date"),
        columns=[f"{u}->{v}" for u, v in pair_names],
    )
    if out.isna().any().any():
        raise ValueError("NaN in GNN scores — refusing to cache")
    out.to_parquet(scores_path(cfg))
    return out


def load_scores(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_parquet(scores_path(cfg))
