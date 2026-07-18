"""Phase 4 gate: state.h5 integrity (skipped until scripts/build_state.py ran)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.data_pipeline.state_builder import StateCache, state_path

cfg = load_config()
needs_state = pytest.mark.skipif(
    not state_path(cfg).exists(), reason="run scripts/build_state.py first"
)


@needs_state
def test_state_shapes_and_alignment():
    s = StateCache(cfg)
    T = len(s)
    assert s.features.shape == (T, 15, 6)
    assert s.gnn.shape == (T, 64)
    assert s.sentiment.shape == (T, 15)
    assert s.open.shape == s.close.shape == (T, 15)
    assert len(s.dates) == T and s.dates.is_monotonic_increasing


@needs_state
def test_state_values_sane():
    s = StateCache(cfg)
    for arr in (s.features, s.gnn, s.sentiment, s.open, s.close):
        assert np.isfinite(arr).all()
    assert ((s.gnn >= 0) & (s.gnn <= 1)).all()
    assert ((s.sentiment >= -1) & (s.sentiment <= 1)).all()
    assert (s.open > 0).all() and (s.close > 0).all()


@needs_state
def test_state_covers_rl_window_only():
    s = StateCache(cfg)
    assert s.dates.min() >= np.datetime64(cfg.data.start_date)
    assert s.dates.max() <= np.datetime64(cfg.data.end_date)
    # walk-forward split must leave a usable test period
    assert (s.dates >= np.datetime64(cfg.rl.test_start)).sum() > 20
