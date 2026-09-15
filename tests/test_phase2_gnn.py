"""Phase 2 gate: cascade-label semantics, split hygiene, model determinism."""

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.seeding import set_global_seed
from src.models.gnn.labels import build_samples, temporal_split
from src.models.gnn.model import PropagationGNN

cfg = load_config()


def _toy_shocks() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    shocks = pd.DataFrame(False, index=dates, columns=["A.NS", "B.NS"])
    shocks.loc[dates[0], "A.NS"] = True   # A shocks day 0
    shocks.loc[dates[3], "B.NS"] = True   # B shocks day 3 (inside 5-day window)
    shocks.loc[dates[9], "A.NS"] = True   # A shocks in the truncated tail
    return shocks


def test_cascade_label_semantics():
    samples = build_samples(_toy_shocks(), [("A.NS", "B.NS")], cfg)
    # day-9 shock must be excluded (label window would be truncated)
    assert len(samples) == 1
    s = samples[0]
    assert s.src == "A.NS" and s.dst == "B.NS" and s.label == 1


def test_cascade_negative_label():
    shocks = _toy_shocks()
    shocks.loc[:, "B.NS"] = False  # remove B's shock -> no cascade
    samples = build_samples(shocks, [("A.NS", "B.NS")], cfg)
    assert len(samples) == 1 and samples[0].label == 0


def test_temporal_split_no_overlap_and_gap():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    shocks = pd.DataFrame(True, index=dates, columns=["A.NS"])
    shocks["B.NS"] = False
    samples = build_samples(shocks, [("A.NS", "B.NS")], cfg)
    train, val = temporal_split(samples, cfg)
    last_train = max(s.date for s in train)
    first_val = min(s.date for s in val)
    assert last_train < first_val
    # leakage guard: k business days must separate the periods
    gap_days = len(pd.bdate_range(last_train, first_val)) - 2
    assert gap_days >= cfg.gnn.cascade_window_days


def test_model_forward_shape_and_determinism():
    set_global_seed(cfg.project.seed)
    m1 = PropagationGNN(in_dim=6, cfg=cfg)
    set_global_seed(cfg.project.seed)
    m2 = PropagationGNN(in_dim=6, cfg=cfg)

    x_window = torch.randn(cfg.gnn.temporal_window_days, 15, 6)
    ei = torch.randint(0, 15, (2, 64))
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])
    m1.eval(); m2.eval()
    with torch.no_grad():
        out1, out2 = m1(x_window, ei, pairs), m2(x_window, ei, pairs)
    assert out1.shape == (3,)
    assert torch.equal(out1, out2), "identical seed must give identical weights"


def test_window_pads_and_never_looks_ahead():
    from src.models.gnn.graph_dataset import window_features

    x = torch.arange(5 * 3 * 2, dtype=torch.float32).reshape(5, 3, 2)
    w = window_features(x, day_idx=1, width=4)  # only days 0,1 exist -> 2 pad rows
    assert w.shape == (4, 3, 2)
    assert torch.equal(w[:2], torch.zeros(2, 3, 2))
    assert torch.equal(w[2:], x[:2])
    # never includes a day after day_idx
    w_full = window_features(x, day_idx=4, width=3)
    assert torch.equal(w_full, x[2:5])


def test_saved_weights_load():
    wdir = ROOT / "src" / "models" / "gnn" / "weights"
    if not (wdir / "propagation_gnn.pt").exists():
        pytest.skip("train the GNN first (scripts/train_gnn.py)")
    import json
    state = torch.load(wdir / "propagation_gnn.pt", weights_only=True)
    meta = json.loads((wdir / "train_meta.json").read_text())
    model = PropagationGNN(in_dim=meta["in_dim"], cfg=cfg, pair_dim=meta.get("pair_dim", 0),
                           ctx_dim=meta.get("ctx_dim", 0))
    model.load_state_dict(state)  # raises on any shape/config mismatch
