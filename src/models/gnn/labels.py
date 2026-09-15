"""Shock events and cascade labels for GNN supervision.

A *shock* at node A on day t: |1-day log return| exceeds
max(shock_z * previous-day 21d rolling vol, shock_min_move) — volatility-
scaled so every ticker contributes events at a comparable rate, floored so
calm regimes don't flag economically meaningless moves.

With gnn.shock_basis="excess" the return is measured against the sector
(stock return minus Nifty Auto) and the threshold uses that excess series'
own volatility, and gnn.shock_vol_window sets the sigma window.

Why both exist: with raw returns and a 21d sigma, previous-day volatility
alone ranked pairs at 0.64 AUC out-of-sample (2023H2-2025H2) — better than
any GNN. A 21d sigma mean-reverts within the 5-day label window, so a calm
stock clears 2.5x its stale sigma as soon as normal volatility returns, and
market-wide days fire "cascades" on every calm name at once. A 63d sigma
brings that shortcut to ~0.52 AUC; excess returns remove the common market
move — the part a supply-chain graph cannot claim to explain.
A training sample is a (day t, directed edge A->B) pair where A shocked at t;
its binary label: did B shock on any day in (t, t+k]?

Labels look into the future BY DESIGN — they are supervised targets, computed
once offline. No-lookahead applies to model INPUTS (features at t use data
<= t) and to the temporal train/val split, which prevents label windows from
crossing the split boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.common.config import Config, load_config


def shock_matrix(features: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Boolean (dates x tickers) matrix of shock days."""
    cfg = cfg or load_config()
    w = cfg.gnn.shock_vol_window or cfg.features.volatility_window
    if cfg.gnn.shock_basis == "excess":
        if "excess_ret_1" not in features.columns:
            raise ValueError("gnn.shock_basis='excess' needs the excess_ret_1 column — "
                             "run scripts/fetch_market_data.py and set gnn.history_start")
        ret1 = features["excess_ret_1"].unstack("ticker")
        sigma = ret1.rolling(w).std()
    else:
        ret1 = features["logret_1"].unstack("ticker")
        vol_col = f"vol_{w}"
        sigma = features[vol_col].unstack("ticker") if vol_col in features.columns else ret1.rolling(w).std()
    # previous-day sigma: the shock's own return must not inflate its threshold
    sigma_prev = sigma.shift(1)
    threshold = (cfg.gnn.shock_z * sigma_prev).clip(lower=cfg.gnn.shock_min_move)
    return ret1.abs() > threshold  # NaN sigma on day 0 -> NaN threshold -> False


@dataclass(frozen=True)
class CascadeSample:
    date: pd.Timestamp
    src: str
    dst: str
    label: int


def build_samples(
    shocks: pd.DataFrame,
    directed_edges: list[tuple[str, str]],
    cfg: Config | None = None,
) -> list[CascadeSample]:
    cfg = cfg or load_config()
    k = cfg.gnn.cascade_window_days
    dates = shocks.index
    samples: list[CascadeSample] = []
    # last k dates have truncated label windows — exclude them entirely
    for i, date in enumerate(dates[: len(dates) - k]):
        row = shocks.iloc[i]
        if not row.any():
            continue
        window = shocks.iloc[i + 1 : i + 1 + k]
        for src, dst in directed_edges:
            if row[src]:
                samples.append(
                    CascadeSample(date, src, dst, int(window[dst].any()))
                )
    return samples


def temporal_split(
    samples: list[CascadeSample], cfg: Config | None = None
) -> tuple[list[CascadeSample], list[CascadeSample]]:
    """Chronological split; val = tail fraction of distinct sample dates.

    A gap of ``cascade_window_days`` distinct dates is dropped at the boundary
    so no train label window overlaps the val period (leakage guard).
    """
    cfg = cfg or load_config()
    dates = sorted({s.date for s in samples})
    cut = int(len(dates) * (1 - cfg.gnn.val_tail_fraction))
    k = cfg.gnn.cascade_window_days
    train_dates = set(dates[: max(cut - k, 1)])
    val_dates = set(dates[cut:])
    train = [s for s in samples if s.date in train_dates]
    val = [s for s in samples if s.date in val_dates]
    return train, val
