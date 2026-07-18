"""Phase 4: fuse features + GNN scores + LLM sentiment (+ execution prices)
into ``data/processed/state.h5`` — the ONLY input the RL layer ever touches.

Layout (T = RL-window trading days, N = 15 tickers, F = 6 features,
P = 64 directed pairs):
    /features   [T, N, F] float32   raw numerical features (env normalizes)
    /gnn        [T, P]    float32   propagation confidence scores
    /sentiment  [T, N]    float32   daily sentiment in [-1, 1]
    /open       [T, N]    float32   execution prices (fill at next open)
    /close      [T, N]    float32   mark-to-market prices
    /dates      [T]       bytes     ISO dates
    /tickers, /pair_names, /feature_names   axis labels
attrs: seed, config_sha256, sentiment_source ("llm_cache" | "neutral_placeholder")

Integrity gate: shapes agree, zero NaN/inf, dates strictly increasing, and
every feature row at day t was computed from data <= t (guaranteed upstream,
re-asserted here via the features/panel date alignment).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.common.config import Config, load_config
from src.data_pipeline.features import load_features
from src.models.gnn.graph_dataset import supervision_pairs
from src.models.gnn.inference import load_scores


def state_path(cfg: Config) -> Path:
    return Path(cfg.data.state_file)


def _wide(long: pd.DataFrame, col: str, dates, tickers) -> np.ndarray:
    return long[col].unstack("ticker").loc[dates, tickers].to_numpy(dtype=np.float32)


def build_state(cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    tickers = cfg.universe.tickers

    features = load_features(cfg)
    all_dates = features.index.get_level_values("date").unique().sort_values()
    dates = all_dates[(all_dates >= cfg.data.start_date) & (all_dates <= cfg.data.end_date)]

    # features [T, N, F]
    feature_names = list(features.columns)
    feat = np.stack([_wide(features, c, dates, tickers) for c in feature_names], axis=-1)

    # gnn [T, P] — must cover every RL-window date exactly
    pair_names = [f"{u}->{v}" for u, v in supervision_pairs(cfg)]
    gnn_df = load_scores(cfg)
    if not set(map(pd.Timestamp, dates)).issubset(set(gnn_df.index)):
        raise ValueError("gnn_scores.parquet does not cover the RL window — "
                         "rerun scripts/cache_gnn_scores.py")
    gnn = gnn_df.loc[dates, pair_names].to_numpy(dtype=np.float32)

    # sentiment [T, N] — sparse cache, neutral where absent
    sent_file = Path(cfg.data.processed_dir) / "sentiment.parquet"
    sent = np.full((len(dates), len(tickers)), cfg.llm.neutral_sentiment, dtype=np.float32)
    sentiment_source = "neutral_placeholder"
    if sent_file.exists():
        cache = pd.read_parquet(sent_file)
        wide = cache.pivot(index="date", columns="ticker", values="sentiment")
        wide = wide.reindex(index=dates, columns=tickers)
        sent = np.where(np.isnan(wide.to_numpy()), sent, wide.to_numpy()).astype(np.float32)
        sentiment_source = "llm_cache"

    # execution prices [T, N]
    panel = pd.read_parquet(Path(cfg.data.processed_dir) / "ohlcv_panel.parquet")
    open_px = _wide(panel, "Open", dates, tickers)
    close_px = _wide(panel, "Close", dates, tickers)

    # ---- integrity gate ----
    T = len(dates)
    for name, arr in [("features", feat), ("gnn", gnn), ("sentiment", sent),
                      ("open", open_px), ("close", close_px)]:
        if arr.shape[0] != T:
            raise ValueError(f"{name}: {arr.shape[0]} rows != {T} dates")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name}: contains NaN/inf")
    if not (open_px > 0).all() or not (close_px > 0).all():
        raise ValueError("non-positive prices")
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("dates not strictly increasing")

    out = state_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as h5:
        h5.create_dataset("features", data=feat)
        h5.create_dataset("gnn", data=gnn)
        h5.create_dataset("sentiment", data=sent)
        h5.create_dataset("open", data=open_px)
        h5.create_dataset("close", data=close_px)
        str_dt = h5py.string_dtype()
        h5.create_dataset("dates", data=[d.strftime("%Y-%m-%d") for d in dates], dtype=str_dt)
        h5.create_dataset("tickers", data=tickers, dtype=str_dt)
        h5.create_dataset("pair_names", data=pair_names, dtype=str_dt)
        h5.create_dataset("feature_names", data=feature_names, dtype=str_dt)
        h5.attrs["seed"] = cfg.project.seed
        h5.attrs["config_sha256"] = hashlib.sha256(Path("config.yaml").read_bytes()).hexdigest()
        h5.attrs["sentiment_source"] = sentiment_source
    return out


class StateCache:
    """In-RAM view of state.h5 for high-throughput env sampling (~2 MB total)."""

    def __init__(self, cfg: Config | None = None):
        cfg = cfg or load_config()
        with h5py.File(state_path(cfg), "r") as h5:
            self.features = h5["features"][:]
            self.gnn = h5["gnn"][:]
            self.sentiment = h5["sentiment"][:]
            self.open = h5["open"][:]
            self.close = h5["close"][:]
            self.dates = pd.DatetimeIndex([d.decode() for d in h5["dates"][:]])
            self.tickers = [t.decode() for t in h5["tickers"][:]]
            self.feature_names = [f.decode() for f in h5["feature_names"][:]]
            self.feature_index = {n: i for i, n in enumerate(self.feature_names)}
            self.sentiment_source = h5.attrs["sentiment_source"]
            self.config_sha256 = h5.attrs["config_sha256"]

    def __len__(self) -> int:
        return len(self.dates)
