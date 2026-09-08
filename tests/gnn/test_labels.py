"""Shock detection and cascade-label construction (pure, synthetic data)."""

import pandas as pd
import pytest

from src.models.gnn.labels import build_samples, shock_matrix, temporal_split
from tests.gnn.conftest import make_features_frame

T = ["A.NS", "B.NS"]


def _frame(cfg, ret_a, vol_a, ret_b=None, vol_b=None):
    n = len(ret_a)
    dates = pd.bdate_range("2024-01-01", periods=n)
    ret_b = ret_b or [0.0] * n
    vol_b = vol_b or [0.01] * n
    vol_col = f"vol_{cfg.features.volatility_window}"
    return make_features_frame(dates, T, {
        "logret_1": {"A.NS": ret_a, "B.NS": ret_b},
        vol_col: {"A.NS": vol_a, "B.NS": vol_b},
    })


def test_shock_threshold_scales_with_previous_day_vol(cfg):
    # prev-day vol 0.01 -> threshold max(2.5*0.01, 0.02) = 0.025
    f = _frame(cfg, ret_a=[0.0, 0.03, 0.02], vol_a=[0.01, 0.01, 0.01])
    s = shock_matrix(f, cfg)
    assert s.iloc[1]["A.NS"]        # 0.03 > 0.025
    assert not s.iloc[2]["A.NS"]    # 0.02 == floor, strict inequality


def test_shock_day_zero_never_flags(cfg):
    f = _frame(cfg, ret_a=[0.5, 0.0], vol_a=[0.01, 0.01])
    s = shock_matrix(f, cfg)
    assert not s.iloc[0]["A.NS"]    # no previous-day vol -> NaN threshold -> False


def test_shock_min_move_floor(cfg):
    # tiny vol -> 2.5*vol far below floor -> floor governs
    f = _frame(cfg, ret_a=[0.0, 0.015, -0.021], vol_a=[0.001, 0.001, 0.001])
    s = shock_matrix(f, cfg)
    assert not s.iloc[1]["A.NS"]
    assert s.iloc[2]["A.NS"]        # abs() handles negative moves


def test_shock_uses_previous_not_same_day_vol(cfg):
    # same-day vol is huge; if it were used the 0.05 move would not flag
    f = _frame(cfg, ret_a=[0.0, 0.05], vol_a=[0.01, 10.0])
    s = shock_matrix(f, cfg)
    assert s.iloc[1]["A.NS"]


def test_shock_matrix_shape_and_dtype(cfg):
    f = _frame(cfg, ret_a=[0.0] * 4, vol_a=[0.01] * 4)
    s = shock_matrix(f, cfg)
    assert s.shape == (4, 2)
    assert s.dtypes.eq(bool).all()


def _shocks(n=12):
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(False, index=dates, columns=T)


def test_label_window_is_exclusive_of_shock_day(cfg):
    s = _shocks()
    s.iloc[0, 0] = True   # A shocks day 0
    s.iloc[0, 1] = True   # B shocks the SAME day -> not a cascade
    out = build_samples(s, [("A.NS", "B.NS")], cfg)
    assert len(out) == 1 and out[0].label == 0


def test_label_window_boundary_inclusive_at_k(cfg):
    k = cfg.gnn.cascade_window_days
    s = _shocks(n=k + 10)
    s.iloc[0, 0] = True
    s.iloc[k, 1] = True   # exactly t+k -> inside (t, t+k]
    out = build_samples(s, [("A.NS", "B.NS")], cfg)
    assert out[0].label == 1

    s.iloc[k, 1] = False
    s.iloc[k + 1, 1] = True   # t+k+1 -> outside
    out = build_samples(s, [("A.NS", "B.NS")], cfg)
    assert out[0].label == 0


def test_only_shocked_source_generates_samples(cfg):
    s = _shocks()
    s.iloc[0, 0] = True   # only A shocks
    out = build_samples(s, [("A.NS", "B.NS"), ("B.NS", "A.NS")], cfg)
    assert [(x.src, x.dst) for x in out] == [("A.NS", "B.NS")]


def test_sample_count_equals_shock_days_times_edges(cfg):
    k = cfg.gnn.cascade_window_days
    s = _shocks(n=20)
    s.loc[:, "A.NS"] = True
    s.loc[:, "B.NS"] = True
    edges = [("A.NS", "B.NS"), ("B.NS", "A.NS")]
    out = build_samples(s, edges, cfg)
    assert len(out) == (20 - k) * len(edges)
    assert all(x.label == 1 for x in out)


def test_truncated_tail_excluded(cfg):
    k = cfg.gnn.cascade_window_days
    s = _shocks(n=k + 3)
    s.iloc[-1, 0] = True   # shock inside the last k days -> no full window
    assert build_samples(s, [("A.NS", "B.NS")], cfg) == []


def test_samples_are_hashable_and_frozen(cfg):
    s = _shocks()
    s.iloc[0, 0] = True
    (x,) = build_samples(s, [("A.NS", "B.NS")], cfg)
    assert len({x, x}) == 1
    with pytest.raises(Exception):
        x.label = 5  # type: ignore[misc]


def test_temporal_split_fraction_and_gap(cfg):
    k = cfg.gnn.cascade_window_days
    n = 100
    s = _shocks(n=n + k)
    s.loc[:, "A.NS"] = True
    out = build_samples(s, [("A.NS", "B.NS")], cfg)
    train, val = temporal_split(out, cfg)
    dates = sorted({x.date for x in out})
    cut = int(len(dates) * (1 - cfg.gnn.val_tail_fraction))
    assert len({x.date for x in val}) == len(dates) - cut
    assert len({x.date for x in train}) == cut - k
    assert max(x.date for x in train) < min(x.date for x in val)
    assert len(train) + len(val) == len(out) - k   # exactly k dates dropped


def test_temporal_split_never_shuffles(cfg):
    s = _shocks(n=40)
    s.loc[:, "A.NS"] = True
    out = build_samples(s, [("A.NS", "B.NS")], cfg)
    train, val = temporal_split(out, cfg)
    assert [x.date for x in train] == sorted(x.date for x in train)
    assert [x.date for x in val] == sorted(x.date for x in val)
