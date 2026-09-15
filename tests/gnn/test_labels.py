"""Shock detection and cascade-label construction (pure, synthetic data)."""

import pandas as pd
import pytest

from src.models.gnn.labels import build_samples, shock_matrix, temporal_split
from tests.gnn.conftest import make_features_frame

T = ["A.NS", "B.NS"]


def _with(cfg, **gnn):
    from src.common.config import Config
    raw = cfg.model_dump()
    raw["gnn"].update(gnn)
    return Config.model_validate(raw)


@pytest.fixture
def raw_cfg(cfg):
    """Legacy shock rule (raw returns, sigma from the features' vol column) —
    the threshold mechanics below are basis-independent."""
    return _with(cfg, shock_basis="raw", shock_vol_window=None)


def _frame(cfg, ret_a, vol_a, ret_b=None, vol_b=None, excess_a=None):
    n = len(ret_a)
    dates = pd.bdate_range("2024-01-01", periods=n)
    ret_b = ret_b or [0.0] * n
    vol_b = vol_b or [0.01] * n
    vol_col = f"vol_{cfg.features.volatility_window}"
    cols = {
        "logret_1": {"A.NS": ret_a, "B.NS": ret_b},
        vol_col: {"A.NS": vol_a, "B.NS": vol_b},
    }
    if excess_a is not None:
        cols["excess_ret_1"] = {"A.NS": excess_a, "B.NS": [0.0] * n}
    return make_features_frame(dates, T, cols)


def test_shock_threshold_scales_with_previous_day_vol(raw_cfg):
    cfg = raw_cfg
    # prev-day vol 0.01 -> threshold max(2.5*0.01, 0.02) = 0.025
    f = _frame(cfg, ret_a=[0.0, 0.03, 0.02], vol_a=[0.01, 0.01, 0.01])
    s = shock_matrix(f, cfg)
    assert s.iloc[1]["A.NS"]        # 0.03 > 0.025
    assert not s.iloc[2]["A.NS"]    # 0.02 == floor, strict inequality


def test_shock_day_zero_never_flags(raw_cfg):
    cfg = raw_cfg
    f = _frame(cfg, ret_a=[0.5, 0.0], vol_a=[0.01, 0.01])
    s = shock_matrix(f, cfg)
    assert not s.iloc[0]["A.NS"]    # no previous-day vol -> NaN threshold -> False


def test_shock_min_move_floor(raw_cfg):
    cfg = raw_cfg
    # tiny vol -> 2.5*vol far below floor -> floor governs
    f = _frame(cfg, ret_a=[0.0, 0.015, -0.021], vol_a=[0.001, 0.001, 0.001])
    s = shock_matrix(f, cfg)
    assert not s.iloc[1]["A.NS"]
    assert s.iloc[2]["A.NS"]        # abs() handles negative moves


def test_shock_uses_previous_not_same_day_vol(raw_cfg):
    cfg = raw_cfg
    # same-day vol is huge; if it were used the 0.05 move would not flag
    f = _frame(cfg, ret_a=[0.0, 0.05], vol_a=[0.01, 10.0])
    s = shock_matrix(f, cfg)
    assert s.iloc[1]["A.NS"]


def test_shock_matrix_shape_and_dtype(raw_cfg):
    cfg = raw_cfg
    f = _frame(cfg, ret_a=[0.0] * 4, vol_a=[0.01] * 4)
    s = shock_matrix(f, cfg)
    assert s.shape == (4, 2)
    assert s.dtypes.eq(bool).all()


def test_shock_vol_window_overrides_feature_window(raw_cfg):
    """shock_vol_window != features window -> sigma recomputed from logret_1."""
    c = _with(raw_cfg, shock_vol_window=2)
    # rolling(2).std of [-0.01, 0.01] = 0.01414; prev-day for day 3 -> thr = 2.5*0.01414 = 0.0354
    f = _frame(c, ret_a=[0.01, -0.01, 0.01, 0.05], vol_a=[10.0] * 4)     # vol_21 column must be ignored
    s = shock_matrix(f, c)
    assert s.iloc[3]["A.NS"]
    f2 = _frame(c, ret_a=[0.01, -0.01, 0.01, 0.03], vol_a=[10.0] * 4)
    assert not shock_matrix(f2, c).iloc[3]["A.NS"]


def test_excess_basis_uses_sector_excess_return_and_its_own_sigma(cfg):
    c = _with(cfg, shock_basis="excess", shock_vol_window=2)
    # raw return is huge every day (market-wide move); excess is flat except day 3
    f = _frame(c, ret_a=[0.08, -0.08, 0.08, 0.08], vol_a=[0.001] * 4,
               excess_a=[0.001, -0.001, 0.001, 0.03])
    s = shock_matrix(f, c)
    assert s["A.NS"].tolist() == [False, False, False, True]


def test_excess_basis_requires_excess_column(cfg):
    c = _with(cfg, shock_basis="excess")
    f = _frame(c, ret_a=[0.0, 0.03], vol_a=[0.01, 0.01])
    with pytest.raises(ValueError, match="excess_ret_1"):
        shock_matrix(f, c)


def test_unknown_shock_basis_rejected(cfg):
    with pytest.raises(ValueError):
        _with(cfg, shock_basis="sector")


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
