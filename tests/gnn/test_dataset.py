"""Tensor bridge: feature tensor layout, windowing, and the real dataset's
shape / normalization / split hygiene."""

import pandas as pd
import pytest
import torch

from src.models.gnn.graph_dataset import (
    GraphData,
    features_tensor,
    supervision_pairs,
    window_features,
)
from tests.gnn.conftest import make_features_frame


def _long_frame():
    dates = pd.bdate_range("2024-01-01", periods=3)
    return dates, make_features_frame(dates, ["A.NS", "B.NS"], {
        "f0": {"A.NS": [1, 2, 3], "B.NS": [10, 20, 30]},
        "f1": {"A.NS": [-1, -2, -3], "B.NS": [-10, -20, -30]},
    })


def test_features_tensor_layout():
    dates, f = _long_frame()
    out_dates, x = features_tensor(f, ["A.NS", "B.NS"])
    assert x.shape == (3, 2, 2)
    assert list(out_dates) == list(dates)
    assert x[1, 0, 0] == 2 and x[1, 1, 0] == 20      # [day, node, feature]
    assert x[2, 1, 1] == -30


def test_features_tensor_respects_ticker_order():
    _, f = _long_frame()
    _, x = features_tensor(f, ["B.NS", "A.NS"])
    assert x[0, 0, 0] == 10 and x[0, 1, 0] == 1


def test_window_features_full_and_padded():
    x = torch.arange(6 * 2 * 1, dtype=torch.float32).reshape(6, 2, 1)
    assert torch.equal(window_features(x, 5, 3), x[3:6])
    w = window_features(x, 0, 3)
    assert w.shape == (3, 2, 1)
    assert torch.equal(w[:2], torch.zeros(2, 2, 1)) and torch.equal(w[2], x[0])


def test_window_width_one_is_the_day_itself():
    x = torch.randn(4, 3, 2)
    assert torch.equal(window_features(x, 2, 1), x[2:3])


def test_graphdata_window_agrees_with_standalone():
    x = torch.randn(7, 3, 2)
    dates = list(pd.bdate_range("2024-01-01", periods=7))
    gd = GraphData(
        tickers=["a", "b", "c"], dates=dates, x=x,
        edge_index=torch.zeros(2, 0, dtype=torch.long), edge_weight=torch.zeros(0),
        directed_pairs=[], train=[], val=[],
        norm_mean=torch.zeros(2), norm_std=torch.ones(2),
    )
    for d in range(7):
        assert torch.equal(gd.window(d, 4), window_features(x, d, 4))
    assert gd.day_index(dates[3]) == 3


# ---- real dataset -----------------------------------------------------------

def test_real_shapes(cfg, real_data):
    T, N, F = real_data.x.shape
    assert N == len(cfg.universe.tickers) == 15
    assert F == len(real_data.node_feature_names)
    assert real_data.node_feature_names[:6] == ["logret_1", "logret_5", "logret_21", "vol_21", "mom_63", "volz_21"]
    assert T == len(real_data.dates)
    assert not torch.isnan(real_data.x).any()
    assert torch.isfinite(real_data.x).all()


def test_real_edge_index_symmetric_and_in_range(real_data):
    ei = real_data.edge_index
    N = real_data.x.shape[1]
    assert ei.min() >= 0 and ei.max() < N
    pairs = set(map(tuple, ei.t().tolist()))
    assert all((v, u) in pairs for u, v in pairs), "message passing must be undirected"
    # 32 curated edges symmetrized = 64 message-passing edges; with
    # bidirectional supervision that equals the number of scored pairs
    assert ei.shape[1] == len(real_data.directed_pairs) == 64


def test_real_supervision_pairs_bidirectional(cfg, real_data):
    pairs = supervision_pairs(cfg)
    assert len(pairs) == len(real_data.directed_pairs)
    fwd = pairs[: len(pairs) // 2]
    rev = pairs[len(pairs) // 2:]
    assert [(v, u) for u, v in fwd] == rev
    assert len(set(pairs)) == len(pairs), "duplicate supervision pairs"


def test_real_train_normalization_uses_train_dates_only(real_data):
    train_dates = {s.date for s in real_data.train}
    mask = torch.tensor([d in train_dates for d in real_data.dates])
    f = len(real_data.norm_mean)                       # normalised channels (excludes 'present')
    x = real_data.x[mask][..., :f]
    flat = x[real_data.present[mask]]                  # absent node-days are excluded from the stats
    assert torch.allclose(flat.mean(0), torch.zeros(f), atol=1e-4)
    assert torch.allclose(flat.std(0), torch.ones(f), atol=1e-3)
    assert (real_data.norm_std > 0).all()


def test_real_split_is_chronological_with_gap(cfg, real_data):
    last_train = max(s.date for s in real_data.train)
    first_val = min(s.date for s in real_data.val)
    assert last_train < first_val
    gap_positions = real_data.day_index(first_val) - real_data.day_index(last_train) - 1
    assert gap_positions >= cfg.gnn.cascade_window_days


def test_real_split_counts_match_train_meta(real_data, train_meta):
    assert len(real_data.train) == train_meta["n_train"]
    assert len(real_data.val) == train_meta["n_val"]
    pos_rate = sum(s.label for s in real_data.train) / len(real_data.train)
    assert pos_rate == pytest.approx(train_meta["train_pos_rate"], abs=1e-3)


def test_real_both_classes_present_in_val(real_data):
    labels = {s.label for s in real_data.val}
    assert labels == {0, 1}


def test_real_samples_reference_valid_pairs(cfg, real_data):
    valid = set(supervision_pairs(cfg))
    for s in real_data.train + real_data.val:
        assert (s.src, s.dst) in valid
