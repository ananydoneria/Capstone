"""PropagationGNN architecture invariants (pure, synthetic tensors)."""

import pytest
import torch

from src.common.seeding import set_global_seed
from src.models.gnn.model import PropagationGNN

N, F = 15, 6


def _inputs(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = cfg.gnn.temporal_window_days
    x = torch.randn(W, N, F, generator=g)
    src = torch.randint(0, N, (40,), generator=g)
    dst = torch.randint(0, N, (40,), generator=g)
    ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    pairs = torch.tensor([[0, 1], [1, 0], [2, 3], [7, 12]])
    return x, ei, pairs


def _model(cfg, seed=42):
    set_global_seed(seed)
    m = PropagationGNN(in_dim=F, cfg=cfg)
    m.eval()
    return m


def test_output_shape_and_finite(cfg):
    x, ei, pairs = _inputs(cfg)
    out = _model(cfg)(x, ei, pairs)
    assert out.shape == (len(pairs),)
    assert torch.isfinite(out).all()


def test_seed_determinism(cfg):
    x, ei, pairs = _inputs(cfg)
    assert torch.equal(_model(cfg)(x, ei, pairs), _model(cfg)(x, ei, pairs))


def test_eval_mode_is_dropout_free(cfg):
    x, ei, pairs = _inputs(cfg)
    m = _model(cfg)
    with torch.no_grad():
        assert torch.equal(m(x, ei, pairs), m(x, ei, pairs))


def test_train_mode_dropout_is_stochastic(cfg):
    x, ei, pairs = _inputs(cfg)
    m = _model(cfg)
    m.train()
    torch.manual_seed(1)
    a = m(x, ei, pairs)
    torch.manual_seed(2)
    b = m(x, ei, pairs)
    assert not torch.equal(a, b)


def test_all_parameters_receive_gradients(cfg):
    x, ei, pairs = _inputs(cfg)
    m = _model(cfg)
    m.train()
    loss = m(x, ei, pairs).pow(2).mean()
    loss.backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert dead == [], f"parameters without gradient: {dead}"


def test_pair_scores_independent_of_batching(cfg):
    x, ei, pairs = _inputs(cfg)
    m = _model(cfg)
    with torch.no_grad():
        joint = m(x, ei, pairs)
        single = torch.cat([m(x, ei, pairs[i : i + 1]) for i in range(len(pairs))])
    assert torch.allclose(joint, single, atol=1e-6)


def test_direction_matters(cfg):
    x, ei, pairs = _inputs(cfg)
    m = _model(cfg)
    with torch.no_grad():
        out = m(x, ei, pairs)
    assert out[0] != out[1], "(0,1) and (1,0) must be scored differently"


def test_node_permutation_equivariance(cfg):
    """Relabelling nodes consistently in x, edge_index and pairs must not
    change the pair logits (GraphSAGE + per-node GRU are permutation-equivariant)."""
    x, ei, pairs = _inputs(cfg)
    m = _model(cfg)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(3))
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(N)
    with torch.no_grad():
        ref = m(x, ei, pairs)
        out = m(x[:, perm], inv[ei], inv[pairs])
    assert torch.allclose(ref, out, atol=1e-5)


def test_temporal_order_matters(cfg):
    x, ei, pairs = _inputs(cfg)
    m = _model(cfg)
    with torch.no_grad():
        a = m(x, ei, pairs)
        b = m(x.flip(0), ei, pairs)
    assert not torch.allclose(a, b), "GRU rollup must be order-sensitive"


def test_graph_structure_matters(cfg):
    x, ei, pairs = _inputs(cfg)
    m = _model(cfg)
    empty = torch.zeros(2, 0, dtype=torch.long)
    with torch.no_grad():
        a = m(x, ei, pairs)
        b = m(x, empty, pairs)
    assert not torch.allclose(a, b), "removing all edges must change scores"


def test_zero_padded_window_is_valid_input(cfg):
    x, ei, pairs = _inputs(cfg)
    x[:5] = 0.0
    out = _model(cfg)(x, ei, pairs)
    assert torch.isfinite(out).all()


def test_gat_variant_builds_and_runs(cfg):
    gat_cfg = cfg.model_copy(deep=True)
    gat_cfg.gnn.arch = "gat"
    x, ei, pairs = _inputs(cfg)
    m = _model(gat_cfg)
    out = m(x, ei, pairs)
    assert out.shape == (len(pairs),) and torch.isfinite(out).all()


def test_unknown_arch_rejected(cfg):
    bad = cfg.model_copy(deep=True)
    bad.gnn.arch = "transformer"
    with pytest.raises(KeyError):
        PropagationGNN(in_dim=F, cfg=bad)


def test_parameter_count_matches_config(cfg):
    m = _model(cfg)
    g = cfg.gnn
    n = sum(p.numel() for p in m.parameters())
    # SAGE: 2 linears (lin_l w/ bias, lin_r no bias) per layer
    sage = sum(2 * din * g.hidden_dim + g.hidden_dim
               for din in [F] + [g.hidden_dim] * (g.num_layers - 1))
    gru = 3 * (g.hidden_dim * g.temporal_hidden_dim + g.temporal_hidden_dim ** 2
               + 2 * g.temporal_hidden_dim)
    head = (2 * g.temporal_hidden_dim * g.temporal_hidden_dim + g.temporal_hidden_dim
            + g.temporal_hidden_dim + 1)
    assert n == sage + gru + head
