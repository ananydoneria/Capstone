"""Per-node numerical features for the GNN and the RL state vector.

Every feature at day t is computed exclusively from data <= t (no-lookahead
guardrail). Output: long-format DataFrame indexed by (date, ticker) with one
column per feature, saved to ``data/processed/features.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import Config, load_config


def _wide(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    return panel[col].unstack("ticker")


def build_features(panel: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    f = cfg.features
    close = _wide(panel, "Close")
    volume = _wide(panel, "Volume")

    feats: dict[str, pd.DataFrame] = {}
    for w in f.return_windows:
        feats[f"logret_{w}"] = np.log(close / close.shift(w))
    feats[f"vol_{f.volatility_window}"] = (
        feats["logret_1"].rolling(f.volatility_window).std()
    )
    feats[f"mom_{f.momentum_window}"] = np.log(close / close.shift(f.momentum_window))
    vz_w = f.volume_zscore_window
    vol_mean = volume.rolling(vz_w).mean()
    vol_std = volume.rolling(vz_w).std()
    feats[f"volz_{vz_w}"] = ((volume - vol_mean) / vol_std).replace(
        [np.inf, -np.inf], np.nan
    )

    long = pd.concat(
        {name: df.stack() for name, df in feats.items()}, axis=1
    )
    long.index.names = ["date", "ticker"]

    # Warmup: drop dates before every feature is defined for every ticker
    warmup = max(max(f.return_windows), f.momentum_window, f.volatility_window, vz_w)
    valid_dates = close.index[warmup:]
    long = long.loc[long.index.get_level_values("date").isin(valid_dates)]
    # volz can still be NaN on zero-variance windows (halt stretches) -> neutral 0
    long[f"volz_{vz_w}"] = long[f"volz_{vz_w}"].fillna(0.0)

    if long.isna().any().any():
        bad = long.columns[long.isna().any()].tolist()
        raise ValueError(f"NaNs after warmup in features: {bad}")
    return long.sort_index()


def save_features(features: pd.DataFrame, cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    out = Path(cfg.data.processed_dir) / "features.parquet"
    features.to_parquet(out)
    return out


def load_features(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_parquet(Path(cfg.data.processed_dir) / "features.parquet")
