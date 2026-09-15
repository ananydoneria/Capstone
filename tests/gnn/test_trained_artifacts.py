"""Integration tests on the trained GNN: weights/meta consistency, metric
reproduction, baseline comparison, cached-score integrity, no-lookahead.
All skip when artifacts are absent."""

import pandas as pd
import pytest
import torch

from src.models.gnn.graph_dataset import supervision_pairs
from src.models.gnn.train import WEIGHTS_DIR, _batch_by_day, _epoch_scores, auc_score, forward_all_pairs, make_model


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


def test_state_dict_shapes_match_config(cfg, train_meta, real_data):
    state = torch.load(WEIGHTS_DIR / "propagation_gnn.pt", weights_only=True)
    assert train_meta["in_dim"] == real_data.x.shape[-1]
    assert train_meta.get("pair_dim", 0) == real_data.pair_dim
    assert train_meta.get("ctx_dim", 0) == real_data.ctx_dim
    fresh = make_model(cfg, real_data).state_dict()
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
    labels = torch.tensor([s.label for s in real_data.val], dtype=torch.float32)

    ret1 = real_data.ret1
    tr_dates = sorted({s.date for s in real_data.train})
    corr = ret1.loc[tr_dates[0]: tr_dates[-1]].corr()
    corr_auc = auc_score(labels, torch.tensor([corr.loc[s.src, s.dst] for s in real_data.val]))

    vol = real_data.vol.shift(1)
    vol_auc = auc_score(labels, torch.tensor([vol.loc[s.date, s.dst] for s in real_data.val]))
    # a rule can be flipped for free: an AUC of 0.36 is a 0.64 rule the other way round
    vol_rule = max(vol_auc, 1 - vol_auc)

    gnn_auc, _, _ = _val_auc(cfg, trained_model, real_data)
    assert gnn_auc > vol_rule, f"GNN {gnn_auc:.3f} <= dst-vol rule {vol_rule:.3f} (raw {vol_auc:.3f})"
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
    W = cfg.gnn.temporal_window_days
    days = [0, 5, 100, 500, len(real_data.dates) - 1]
    with torch.no_grad():
        for d in days:
            fresh = torch.sigmoid(forward_all_pairs(trained_model, real_data, d, W))
            cached = torch.tensor(cached_scores.iloc[d].to_numpy(), dtype=torch.float32)
            assert torch.allclose(fresh, cached, atol=1e-5), f"day {d} differs"


def test_inference_has_no_lookahead(cfg, real_data, trained_model):
    """Perturbing every input AFTER day t (node features, market context, pair
    features) must leave day t's scores unchanged."""
    import copy
    W = cfg.gnn.temporal_window_days
    t = 300
    with torch.no_grad():
        ref = forward_all_pairs(trained_model, real_data, t, W)
        noisy = copy.copy(real_data)
        noisy.x = real_data.x.clone(); noisy.x[t + 1:] = torch.randn_like(noisy.x[t + 1:]) * 10
        if real_data.ctx is not None:
            noisy.ctx = real_data.ctx.clone(); noisy.ctx[t + 1:] = torch.randn_like(noisy.ctx[t + 1:]) * 10
        if real_data.pair_feat is not None:
            noisy.pair_feat = real_data.pair_feat.clone()
            noisy.pair_feat[t + 1:] = torch.randn_like(noisy.pair_feat[t + 1:])
        out = forward_all_pairs(trained_model, noisy, t, W)
    assert torch.equal(ref, out)


def test_inference_depends_on_recent_history(cfg, real_data, trained_model):
    """Sanity: the temporal window is actually used — perturbing a day inside
    the window changes the output."""
    W = cfg.gnn.temporal_window_days
    t = 300
    x = real_data.x.clone()
    with torch.no_grad():
        ref = forward_all_pairs(trained_model, real_data, t, W)
        x[t - 3] += 5.0
        out = forward_all_pairs(trained_model, real_data, t, W, x=x)
    assert not torch.allclose(ref, out)
