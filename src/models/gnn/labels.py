"""Shock events and cascade labels for GNN supervision.

A *shock* at node A on day t: |1-day log return| > threshold.
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
    """Boolean (dates x tickers) matrix of shock days from the logret_1 feature."""
    cfg = cfg or load_config()
    ret1 = features["logret_1"].unstack("ticker")
    return ret1.abs() > cfg.gnn.shock_return_threshold


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
