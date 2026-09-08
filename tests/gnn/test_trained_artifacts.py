"""Integration tests on the trained GNN: weights/meta consistency, metric
reproduction, baseline comparison, cached-score integrity, no-lookahead.
All skip when artifacts are absent."""

import pandas as pd
import pytest
import torch

from src.data_pipeline.features import load_features
from src.models.gnn.graph_dataset import supervision_pairs, window_features
from src.models.gnn.model import PropagationGNN
from src.models.gnn.train import WEIGHTS_DIR, _batch_by_day, _epoch_scores, auc_score


def _val_auc(cfg, model, data):
    with torch.no_grad():
        logits, labels = _epoch_scores(
            model, data, _batch_by_day(data, data.val), cfg.gnn.temporal_window_days
        )
    return auc_score(labels, logits), logits, labels


def test_meta_matches_config(cfg, train_meta, real_data):
    assert train_meta["arch"] == cfg.gnn.arch
    assert train_meta["seed"] == cfg.project.seed
    assert train_meta["in_dim"] == real_data.x.shape[-1]
    for key in ("shock_z", "shock_min_move", "bidirectional_supervision",
                "cascade_window_days", "train_start_date"):
        assert train_meta[key] == getattr(cfg.gnn, key), key


def test_state_dict_shapes_match_config(cfg, train_meta):
    state = torch.load(WEIGHTS_DIR / "propagation_gnn.pt", weights_only=True)
    fresh = PropagationGNN(in_dim=train_meta["in_dim"], cfg=cfg).state_dict()
    assert set(state) == set(fresh)
    for k in state:
        assert state[k].shape == fresh[k].shape, k
        assert torch.isfinite(state[k]).all(), f"non-finite weights in {k}"


def test_persisted_norm_stats_match_dataset(real_data):
    norm = torch.load(WEIGHTS_DIR / "feature_norm.pt", weights_only=True)
    assert torch.allclose(norm["mean"], real_data.norm_mean, atol=1e-6)
    assert torch.allclose(norm["std"], real_data.norm_std, atol=1e-6)


def test_val_auc_reproduces_train_meta(cfg, trained_model, real_data, train_meta):
    auc, _, _ = _val_auc(cfg, trained_model, real_data)
    assert auc == pytest.approx(train_meta["best_val_auc"], abs=0.005)


def test_val_auc_above_chance(cfg, trained_model, real_data):
    auc, _, _ = _val_auc(cfg, trained_model, real_data)
    assert auc > 0.55


def test_gnn_beats_naive_baselines(cfg, trained_model, real_data):
    features = load_features(cfg)
    labels = torch.tensor([s.label for s in real_data.val], dtype=torch.float32)

    ret1 = features["logret_1"].unstack("ticker")
    tr_dates = sorted({s.date for s in real_data.train})
    corr = ret1.loc[tr_dates[0]: tr_dates[-1]].corr()
    corr_auc = auc_score(labels, torch.tensor([corr.loc[s.src, s.dst] for s in real_data.val]))

    vol = features[f"vol_{cfg.features.volatility_window}"].unstack("ticker").shift(1)
    vol_auc = auc_score(labels, torch.tensor([vol.loc[s.date, s.dst] for s in real_data.val]))

    gnn_auc, _, _ = _val_auc(cfg, trained_model, real_data)
    assert gnn_auc > vol_auc, f"GNN {gnn_auc:.3f} <= dst-vol baseline {vol_auc:.3f}"
    assert gnn_auc > corr_auc, f"GNN {gnn_auc:.3f} <= train-corr baseline {corr_auc:.3f}"


def test_train_auc_not_wildly_above_val(cfg, trained_model, real_data):
    """Overfitting guard: the early-stopped checkpoint should not have a
    train/val AUC gap larger than 0.2."""
    with torch.no_grad():
        tl, tlab = _epoch_scores(trained_model, real_data,
                                 _batch_by_day(real_data, real_data.train),
                                 cfg.gnn.temporal_window_days)
    tr_auc = auc_score(tlab, tl)
    va_auc, _, _ = _val_auc(cfg, trained_model, real_data)
    assert tr_auc - va_auc < 0.2, f"train {tr_auc:.3f} vs val {va_auc:.3f}"


def test_scores_are_probabilities_with_spread(cfg, trained_model, real_data):
    _, logits, _ = _val_auc(cfg, trained_model, real_data)
    p = torch.sigmoid(logits)
    assert (p >= 0).all() and (p <= 1).all()
    assert p.std() > 0.02, "near-constant scores carry no signal"


def test_cached_scores_integrity(cfg, cached_scores, real_data):
    pairs = [f"{u}->{v}" for u, v in supervision_pairs(cfg)]
    assert list(cached_scores.columns) == pairs
    assert len(cached_scores) == len(real_data.dates)
    assert cached_scores.index.equals(pd.DatetimeIndex(real_data.dates, name="date"))
    assert not cached_scores.isna().any().any()
    assert ((cached_scores >= 0) & (cached_scores <= 1)).all().all()
    assert cached_scores.index.is_monotonic_increasing


def test_cached_scores_reproducible_from_weights(cfg, cached_scores, real_data, trained_model):
    pos = {t: i for i, t in enumerate(real_data.tickers)}
    pairs = torch.tensor([[pos[u], pos[v]] for u, v in supervision_pairs(cfg)])
    W = cfg.gnn.temporal_window_days
    days = [0, 5, 100, 500, len(real_data.dates) - 1]
    with torch.no_grad():
        for d in days:
            fresh = torch.sigmoid(trained_model(real_data.window(d, W), real_data.edge_index, pairs))
            cached = torch.tensor(cached_scores.iloc[d].to_numpy(), dtype=torch.float32)
            assert torch.allclose(fresh, cached, atol=1e-5), f"day {d} differs"


def test_inference_has_no_lookahead(cfg, real_data, trained_model):
    """Perturbing every feature AFTER day t must leave day t's scores unchanged."""
    pos = {t: i for i, t in enumerate(real_data.tickers)}
    pairs = torch.tensor([[pos[u], pos[v]] for u, v in supervision_pairs(cfg)])
    W = cfg.gnn.temporal_window_days
    t = 300
    x = real_data.x.clone()
    with torch.no_grad():
        ref = trained_model(window_features(x, t, W), real_data.edge_index, pairs)
        x[t + 1:] = torch.randn_like(x[t + 1:]) * 10
        out = trained_model(window_features(x, t, W), real_data.edge_index, pairs)
    assert torch.equal(ref, out)


def test_inference_depends_on_recent_history(cfg, real_data, trained_model):
    """Sanity: the temporal window is actually used — perturbing a day inside
    the window changes the output."""
    pos = {t: i for i, t in enumerate(real_data.tickers)}
    pairs = torch.tensor([[pos[u], pos[v]] for u, v in supervision_pairs(cfg)])
    W = cfg.gnn.temporal_window_days
    t = 300
    x = real_data.x.clone()
    with torch.no_grad():
        ref = trained_model(window_features(x, t, W), real_data.edge_index, pairs)
        x[t - 3] += 5.0
        out = trained_model(window_features(x, t, W), real_data.edge_index, pairs)
    assert not torch.allclose(ref, out)
