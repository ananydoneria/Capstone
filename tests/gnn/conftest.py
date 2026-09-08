"""Shared fixtures for the GNN test suite.

Two tiers:
  * pure/synthetic tests need nothing on disk;
  * integration tests use the real features.parquet + trained weights and
    skip cleanly when those artifacts are absent (they are git-ignored).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.config import Config, load_config  # noqa: E402
from src.common.seeding import set_global_seed  # noqa: E402
from src.models.gnn.graph_dataset import GraphData, build_dataset  # noqa: E402
from src.models.gnn.model import PropagationGNN  # noqa: E402
from src.models.gnn.train import WEIGHTS_DIR  # noqa: E402


def artifacts_present(cfg: Config) -> bool:
    return (
        (WEIGHTS_DIR / "propagation_gnn.pt").exists()
        and (WEIGHTS_DIR / "feature_norm.pt").exists()
        and (WEIGHTS_DIR / "train_meta.json").exists()
        and (Path(cfg.data.processed_dir) / "features.parquet").exists()
    )


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config()


@pytest.fixture(scope="session")
def real_data(cfg: Config) -> GraphData:
    if not artifacts_present(cfg):
        pytest.skip("GNN artifacts missing — run run_phase1.py + train_gnn.py first")
    set_global_seed(cfg.project.seed)
    return build_dataset(cfg)


@pytest.fixture(scope="session")
def train_meta(cfg: Config) -> dict:
    if not artifacts_present(cfg):
        pytest.skip("GNN artifacts missing")
    return json.loads((WEIGHTS_DIR / "train_meta.json").read_text())


@pytest.fixture(scope="session")
def trained_model(cfg: Config, real_data: GraphData) -> PropagationGNN:
    model = PropagationGNN(in_dim=real_data.x.shape[-1], cfg=cfg)
    model.load_state_dict(torch.load(WEIGHTS_DIR / "propagation_gnn.pt", weights_only=True))
    model.eval()
    return model


@pytest.fixture(scope="session")
def cached_scores(cfg: Config) -> pd.DataFrame:
    path = Path(cfg.data.processed_dir) / "gnn_scores.parquet"
    if not path.exists():
        pytest.skip("gnn_scores.parquet missing — run cache_gnn_scores.py first")
    return pd.read_parquet(path)


def make_features_frame(
    dates: pd.DatetimeIndex, tickers: list[str], values: dict[str, dict[str, list[float]]]
) -> pd.DataFrame:
    """Long-format (date, ticker) frame from {feature: {ticker: [per-date values]}}."""
    cols = {}
    for name, per_ticker in values.items():
        wide = pd.DataFrame({t: per_ticker[t] for t in tickers}, index=dates)
        cols[name] = wide.stack()
    out = pd.concat(cols, axis=1)
    out.index.names = ["date", "ticker"]
    return out.sort_index()
