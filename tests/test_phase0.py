"""Phase 0 gate: config loads & validates, universe is correct, seeding is deterministic."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config
from src.common.seeding import set_global_seed


def test_config_loads_and_validates():
    cfg = load_config()
    assert cfg.project.seed == 42
    assert cfg.llm.temperature == 0.0
    assert cfg.data.start_date < cfg.data.end_date
    assert cfg.rl.train_end < cfg.rl.test_start  # walk-forward split ordering


def test_universe_is_15_unique_nse_tickers():
    cfg = load_config()
    tickers = cfg.universe.tickers
    assert len(tickers) == 15
    assert len(set(tickers)) == 15
    assert all(t.endswith(".NS") for t in tickers)


def test_nonzero_llm_temperature_rejected(tmp_path):
    import yaml

    raw = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    raw["llm"]["temperature"] = 0.7
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(Exception, match="temperature"):
        load_config(bad)


def test_seeding_is_deterministic():
    set_global_seed(42)
    a = np.random.default_rng(42).normal(size=8)
    first = np.random.normal(size=8)
    set_global_seed(42)
    b = np.random.default_rng(42).normal(size=8)
    second = np.random.normal(size=8)
    assert np.array_equal(a, b)
    assert np.array_equal(first, second)
