"""Extended GNN inputs: corporate-action adjustment, same-day vs lagged market
context, presence masking, pair features, edge weights, PCA fitted on train
only, and — most important — that none of it can see the future."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_pipeline import market_data as md
from src.models.gnn import graph_dataset as gd
from src.models.gnn.model import PropagationGNN


def _cfg(base, **gnn):
    from src.common.config import Config
    raw = base.model_dump()
    raw["gnn"].update(gnn)
    return Config.model_validate(raw)


def _extended_available(cfg) -> bool:
    p = Path(cfg.data.processed_dir)
    return (p / "gnn_node_features.parquet").exists() and (p / "market_context.parquet").exists()


@pytest.fixture(scope="module")
def ext_cfg(cfg):
    if not _extended_available(cfg):
        pytest.skip("extended data missing — run scripts/fetch_market_data.py")
    return _cfg(cfg, history_start="2011-10-01", market_context="full", breadth=True,
                pair_features=["rolling_corr", "edge_weight", "direction"], arch="graphconv", ctx_pca=8)


# ---------------------------------------------------------------- pure / synthetic

def test_demerger_back_adjustment_neutralises_the_ex_date():
    idx = pd.bdate_range("2025-10-08", periods=5)
    df = pd.DataFrame({"Open": [100, 101, 102, 60, 61], "Close": [101, 102, 100, 61, 62],
                       "Volume": 1.0}, index=idx)
    acts = pd.DataFrame({"ticker": ["X"], "ex_date": [str(idx[3].date())], "factor": ["auto"], "note": [""]})
    adj = md.apply_corporate_actions(df, acts, "X")
    r = np.log(adj["Close"] / adj["Close"].shift(1))
    assert abs(r.iloc[3] - np.log(61 / 60)) < 1e-12      # only the ex-date's own trading remains
    assert np.allclose(r.iloc[1:3], np.log(df["Close"] / df["Close"].shift(1)).iloc[1:3])  # returns before unchanged
    assert (adj.loc[idx[3]:, "Close"] == df.loc[idx[3]:, "Close"]).all()            # post-event untouched


def test_asof_lag_zero_uses_same_day_lag_one_uses_previous_day():
    src = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=pd.DatetimeIndex(
        ["2024-01-01", "2024-01-02", "2024-01-03"], name="src_date"))
    cal = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    assert md._asof(src, cal, lag=0)["v"].tolist() == [2.0, 3.0]
    assert md._asof(src, cal, lag=1)["v"].tolist() == [1.0, 2.0]


def test_asof_marks_stale_values_missing():
    src = pd.DataFrame({"v": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"], name="src_date"))
    out = md._asof(src, pd.DatetimeIndex(["2024-03-01"]), lag=0)
    assert out["v"].isna().all()


def test_series_features_level_vs_price(cfg):
    idx = pd.bdate_range("2024-01-01", periods=30)
    df = pd.DataFrame({"Close": np.linspace(10, 20, 30)}, index=idx)
    price = md._series_features(df, md.Series("X", "x", "india", 0), cfg)
    level = md._series_features(df, md.Series("V", "some_vix", "india", 0, "level"), cfg)
    assert list(price.columns) == ["x_ret1", "x_ret5", "x_vol21"]
    assert list(level.columns) == ["some_vix_level", "some_vix_chg1", "some_vix_chg5"]
    assert np.isclose(level["some_vix_level"].iloc[-1], np.log(20))


def test_graphconv_uses_edge_weights_sage_does_not(cfg):
    torch.manual_seed(0)
    x = torch.randn(cfg.gnn.temporal_window_days, 5, 6)
    ei = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    pairs = torch.tensor([[0, 1], [3, 4]])
    w1, w2 = torch.ones(6), torch.tensor([0.1, 0.1, 1.0, 1.0, 5.0, 5.0])
    for arch, should_change in [("graphconv", True), ("graphsage", False)]:
        torch.manual_seed(1)
        m = PropagationGNN(in_dim=6, cfg=_cfg(cfg, arch=arch)).eval()
        with torch.no_grad():
            a, b = m(x, ei, pairs, edge_weight=w1), m(x, ei, pairs, edge_weight=w2)
        assert (not torch.allclose(a, b)) == should_change, arch


def test_head_accepts_pair_features_and_context(cfg):
    m = PropagationGNN(in_dim=6, cfg=cfg, pair_dim=3, ctx_dim=8).eval()
    x = torch.randn(cfg.gnn.temporal_window_days, 5, 6)
    ei = torch.tensor([[0, 1], [1, 0]])
    pairs = torch.tensor([[0, 1], [1, 0], [2, 3]])
    out = m(x, ei, pairs, pair_feat=torch.randn(3, 3), ctx=torch.randn(8))
    assert out.shape == (3,)
    with torch.no_grad():
        c1 = m(x, ei, pairs, pair_feat=torch.zeros(3, 3), ctx=torch.zeros(8))
        c2 = m(x, ei, pairs, pair_feat=torch.zeros(3, 3), ctx=torch.ones(8))
    assert not torch.allclose(c1, c2), "context must influence the score"


# ---------------------------------------------------------------- real extended data

def test_presence_masking(ext_cfg):
    d = gd.build_dataset(ext_cfg)
    son = d.tickers.index("SONACOMS.NS")
    first = next(i for i in range(len(d.dates)) if d.present[i, son])
    assert d.dates[first] >= pd.Timestamp("2021-06-01")
    assert torch.all(d.x[:first, son, :-1] == 0) and torch.all(d.x[:first, son, -1] == 0)
    assert torch.all(d.x[first:, son, -1] == 1)
    for s in d.train + d.val:
        if "SONACOMS.NS" in (s.src, s.dst):
            assert s.date >= d.dates[first]
    assert d.node_feature_names[-1] == "present"


def test_extended_shapes_and_finiteness(ext_cfg):
    d = gd.build_dataset(ext_cfg)
    assert d.x.shape[1] == 15 and d.ctx.shape == (len(d.dates), 8) and d.pair_feat.shape[1:] == (64, 3)
    for t in (d.x, d.ctx, d.pair_feat):
        assert torch.isfinite(t).all()
    short = gd.build_dataset(_cfg(ext_cfg, history_start="2021-07-01", market_context="none", breadth=False,
                                  pair_features=[], arch="graphsage", ctx_pca=0))   # same label, 2021+ only
    assert len(d.train) > 2.5 * len(short.train), "long history should multiply the training set"


def test_pca_basis_fit_on_train_days_only(ext_cfg):
    raw = gd.raw_inputs(ext_cfg)
    samples = gd.all_samples(ext_cfg, raw)
    train, _ = gd.temporal_split(samples, ext_cfg)
    tr = {s.date for s in train}
    a = gd.compute_stats(raw, tr)
    raw2 = dict(raw)
    ctx = raw["ctx_raw"].clone()
    later = torch.tensor([d not in tr for d in raw["dates"]])
    ctx[later] = torch.randn_like(ctx[later]) * 50       # scramble every non-train day
    raw2["ctx_raw"] = ctx
    b = gd.compute_stats(raw2, tr)
    for k in ("mean", "std", "ctx_mean", "ctx_std", "ctx_pca"):
        assert torch.allclose(a[k], b[k]), k


def test_inference_inputs_match_training_inputs(ext_cfg):
    d = gd.build_dataset(ext_cfg)
    stats = {"mean": d.norm_mean, "std": d.norm_std, "ctx_mean": d.ctx_mean, "ctx_std": d.ctx_std,
             "ctx_names": d.ctx_names, **d.extra_stats}
    e = gd.build_inference_inputs(ext_cfg, stats)
    assert torch.equal(d.x, e.x) and torch.equal(d.ctx, e.ctx) and torch.equal(d.pair_feat, e.pair_feat)


def test_node_features_no_lookahead(ext_cfg):
    uni = md.universe_frames(ext_cfg)
    cut = pd.Timestamp("2019-06-14")
    full = md.build_node_features(ext_cfg, md.build_gnn_panel(ext_cfg, uni))
    trunc = md.build_node_features(ext_cfg, md.build_gnn_panel(ext_cfg, {t: df[df.index <= cut] for t, df in uni.items()}))
    a, b = full.xs(cut, level="date"), trunc.xs(cut, level="date")
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10)


def test_market_context_no_lookahead(ext_cfg):
    """Truncating every source series at day t must not change context at t.
    Lag-1 series additionally must not change when day t's own bar is dropped."""
    cut = pd.Timestamp("2022-03-15")
    macro = {s.ticker: md.load_close(ext_cfg, s.ticker) for s in md.MACRO}
    macro = {k: v for k, v in macro.items() if v is not None}
    baskets = {b: {t: df for t in ts if (df := md.load_close(ext_cfg, t)) is not None}
               for b, ts in md.BASKETS.items()}
    cal = pd.DatetimeIndex(sorted(md.load_node_features(ext_cfg).index.get_level_values("date").unique()))
    cal_cut = cal[cal <= cut]
    full = md.build_market_context(ext_cfg, cal_cut, macro, baskets).loc[cut]
    trunc = md.build_market_context(
        ext_cfg, cal_cut, {k: v[v.index <= cut] for k, v in macro.items()},
        {b: {t: df[df.index <= cut] for t, df in fr.items()} for b, fr in baskets.items()}).loc[cut]
    pd.testing.assert_series_equal(full, trunc, check_exact=False, rtol=1e-10)
    lag1 = {s.ticker for s in md.MACRO if s.lag == 1}
    no_today = md.build_market_context(
        ext_cfg, cal_cut, {k: (v[v.index < cut] if k in lag1 else v[v.index <= cut]) for k, v in macro.items()},
        {b: {t: df[df.index <= cut] for t, df in fr.items()} for b, fr in baskets.items()}).loc[cut]
    pd.testing.assert_series_equal(full, no_today, check_exact=False, rtol=1e-10)


def test_rolling_corr_pair_feature_no_lookahead(ext_cfg):
    raw = gd.raw_inputs(ext_cfg)
    t = 2000
    ret = raw["ret1"].copy()
    ret.iloc[t + 1:] = np.random.default_rng(0).normal(0, 0.5, ret.iloc[t + 1:].shape)
    pf = gd._pair_features(ext_cfg, ret, raw["pair_names"])
    assert torch.equal(pf[t], raw["pair_feat"][t])


def test_labels_never_read_past_label_end_date(cfg):
    """The PPO test window (2026) must not influence GNN training or early
    stopping: every train/val label window ends on or before label_end_date."""
    if not cfg.gnn.label_end_date or not _extended_available(cfg):
        pytest.skip("label_end_date unset or extended data missing")
    d = gd.build_dataset(cfg)
    end = pd.Timestamp(cfg.gnn.label_end_date)
    k = cfg.gnn.cascade_window_days
    last_window_end = max(d.dates[d.day_index(s.date) + k] for s in d.train + d.val)
    assert last_window_end <= end
    assert max(s.date for s in d.train) < min(s.date for s in d.val)
