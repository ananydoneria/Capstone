"""Training helpers: AUC implementation, per-day batching, class weighting,
and an end-to-end overfit smoke test on synthetic data."""

import math

import pandas as pd
import pytest
import torch
from torch import nn

from src.models.gnn.graph_dataset import GraphData
from src.models.gnn.labels import CascadeSample
from src.models.gnn.model import PropagationGNN
from src.models.gnn.train import _batch_by_day, _epoch_scores, auc_score


def test_auc_perfect_inverted_and_ties():
    y = torch.tensor([0, 0, 1, 1.0])
    assert auc_score(y, torch.tensor([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc_score(y, torch.tensor([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc_score(y, torch.tensor([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_auc_single_class_is_nan():
    assert math.isnan(auc_score(torch.tensor([1.0, 1.0]), torch.tensor([0.1, 0.2])))
    assert math.isnan(auc_score(torch.tensor([0.0, 0.0]), torch.tensor([0.1, 0.2])))


def test_auc_is_rank_based():
    y = torch.tensor([0, 1, 0, 1.0])
    s = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert auc_score(y, s) == auc_score(y, s * 1000 - 7)


def test_auc_matches_sklearn_if_available():
    sk = pytest.importorskip("sklearn.metrics")
    g = torch.Generator().manual_seed(0)
    y = (torch.rand(500, generator=g) > 0.7).float()
    s = torch.randn(500, generator=g) + y
    assert auc_score(y, s) == pytest.approx(sk.roc_auc_score(y.numpy(), s.numpy()), abs=1e-6)


def _toy_graphdata(n_days=6):
    dates = list(pd.bdate_range("2024-01-01", periods=n_days))
    return GraphData(
        tickers=["A", "B", "C"], dates=dates, x=torch.zeros(n_days, 3, 2),
        edge_index=torch.zeros(2, 0, dtype=torch.long), edge_weight=torch.zeros(0),
        directed_pairs=[], train=[], val=[],
        norm_mean=torch.zeros(2), norm_std=torch.ones(2),
    )


def test_batch_by_day_groups_and_orders():
    gd = _toy_graphdata()
    d = gd.dates
    samples = [
        CascadeSample(d[4], "A", "B", 1),
        CascadeSample(d[1], "B", "C", 0),
        CascadeSample(d[4], "C", "A", 0),
    ]
    batches = _batch_by_day(gd, samples)
    assert [b[0] for b in batches] == [1, 4]
    day4 = batches[1]
    assert day4[1].tolist() == [[0, 1], [2, 0]]
    assert day4[2].tolist() == [1.0, 0.0]
    assert day4[2].dtype == torch.float32


def test_pos_weight_balances_classes():
    n_tr, n_pos = 2133, 380
    pos_weight = torch.tensor([(n_tr - n_pos) / max(n_pos, 1)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # a positive and a negative at the same logit contribute equal weighted
    # loss mass once scaled by their class frequency
    logit = torch.tensor([0.0])
    lp = loss_fn(logit, torch.tensor([1.0])) * n_pos
    ln = loss_fn(logit, torch.tensor([0.0])) * (n_tr - n_pos)
    assert torch.allclose(lp, ln)


def test_model_can_overfit_synthetic_cascades(cfg):
    """Learning mechanics smoke test: a deterministic rule linking source
    features to labels must be recoverable to AUC > 0.9 within a few dozen
    steps. Guards against silent gradient / batching / shape breakage."""
    torch.manual_seed(0)
    n_days, N, F = 40, 8, 6
    W = cfg.gnn.temporal_window_days
    x = torch.randn(n_days, N, F)
    ring = torch.tensor([[i, (i + 1) % N] for i in range(N)]).t()
    ei = torch.cat([ring, ring.flip(0)], dim=1)
    pairs = ring.t()

    dates = list(pd.bdate_range("2024-01-01", periods=n_days))
    gd = GraphData(
        tickers=[str(i) for i in range(N)], dates=dates, x=x,
        edge_index=ei, edge_weight=torch.ones(ei.shape[1]),
        directed_pairs=pairs.tolist(), train=[], val=[],
        norm_mean=torch.zeros(F), norm_std=torch.ones(F),
    )
    # label = 1 iff source's feature-0 on the window's last day is positive
    batches = [
        (d, pairs, (x[d, pairs[:, 0], 0] > 0).float()) for d in range(W, n_days)
    ]

    model = PropagationGNN(in_dim=F, cfg=cfg)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    first = None
    for _ in range(60):
        model.train()
        total = 0.0
        for d, p, y in batches:
            opt.zero_grad()
            loss = loss_fn(model(gd.window(d, W), ei, p), y)
            loss.backward()
            opt.step()
            total += float(loss.detach())
        first = first if first is not None else total
    model.eval()
    with torch.no_grad():
        logits, labels = _epoch_scores(model, gd, batches, W)
    assert total < first * 0.5, "loss did not decrease"
    assert auc_score(labels, logits) > 0.9
