"""Phase 2.5: full-timeline GNN inference cache.

Loads the trained weights + persisted train-period normalization stats, runs
one forward pass per day over the whole node-feature timeline, and caches
sigmoid Propagation Confidence Scores per directed pair to
``data/processed/gnn_scores.parquet`` (index=date, columns "SRC->DST").

Inputs are built by the same code path as training
(graph_dataset.build_inference_inputs), so legacy and extended configurations
(history, market context, pair features, edge weights) are handled alike.

After this, the GNN is never invoked downstream (compute guardrail).

Run:  python scripts/cache_gnn_scores.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.common.config import Config, load_config
from src.common.seeding import set_global_seed
from src.models.gnn.graph_dataset import build_inference_inputs, supervision_pairs
from src.models.gnn.train import WEIGHTS_DIR, forward_all_pairs, load_stats, make_model


def scores_path(cfg: Config) -> Path:
    return Path(cfg.data.processed_dir) / "gnn_scores.parquet"


def load_trained(cfg: Config, data, weights_dir: Path | None = None):
    model = make_model(cfg, data)
    model.load_state_dict(torch.load((weights_dir or WEIGHTS_DIR) / "propagation_gnn.pt", weights_only=True))
    model.eval()
    return model


def run_inference(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    set_global_seed(cfg.project.seed)
    data = build_inference_inputs(cfg, load_stats())
    model = load_trained(cfg, data)

    window = cfg.gnn.temporal_window_days
    with torch.no_grad():
        scores = torch.stack([
            torch.sigmoid(forward_all_pairs(model, data, t, window)) for t in range(len(data.dates))
        ])

    out = pd.DataFrame(
        scores.numpy(),
        index=pd.DatetimeIndex(data.dates, name="date"),
        columns=[f"{u}->{v}" for u, v in supervision_pairs(cfg)],
    )
    if out.isna().any().any():
        raise ValueError("NaN in GNN scores — refusing to cache")
    out.to_parquet(scores_path(cfg))
    return out


def load_scores(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_parquet(scores_path(cfg))
