"""Phase 1 gate: runs offline against the committed pipeline code and the
downloaded artifacts in data/ (skipped cleanly if data hasn't been pulled)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.data_pipeline import features as feat
from src.data_pipeline import ohlcv_ingest as ingest
from src.data_pipeline.relationships import build_graph, load_edge_list, to_edge_index

cfg = load_config()
PANEL_PATH = ROOT / cfg.data.processed_dir / "ohlcv_panel.parquet"
needs_data = pytest.mark.skipif(not PANEL_PATH.exists(), reason="run scripts/run_phase1.py first")


@needs_data
def test_panel_integrity():
    panel = pd.read_parquet(PANEL_PATH)
    assert not panel[["Open", "High", "Low", "Close"]].isna().any().any()
    assert panel.index.get_level_values("ticker").nunique() == 15
    dates = panel.index.get_level_values("date")
    assert dates.min() >= pd.Timestamp(ingest.effective_start(cfg))
    assert dates.max() <= pd.Timestamp(cfg.data.end_date)


@needs_data
def test_panel_has_no_unadjusted_corporate_actions():
    """Yahoo leaves demergers unadjusted (TMPV 2025-10-14 showed -51%,
    MOTHERSON 2022-01-14 +28%). The env would book those as real P&L, so the
    panel must be back-adjusted via data/raw/corporate_actions.csv."""
    panel = pd.read_parquet(PANEL_PATH)
    close = panel["Close"].unstack("ticker")
    r = np.log(close / close.shift(1))
    assert abs(r.loc["2025-10-14", "TMPV.NS"]) < 0.05
    assert abs(r.loc["2022-01-14", "MOTHERSON.NS"]) < 0.10
    worst = r.abs().stack()
    assert worst.max() < 0.30, f"suspicious move {worst.idxmax()}: {worst.max():.2f} — corporate action?"
    # High/Low stay consistent with Open/Close after the adjustment
    assert (panel["High"] >= panel[["Open", "Close"]].max(axis=1) - 1e-6).all()
    assert (panel["Low"] <= panel[["Open", "Close"]].min(axis=1) + 1e-6).all()


@needs_data
def test_features_no_nan_and_consistent_with_prices():
    features = feat.load_features(cfg)
    assert not features.isna().any().any()
    # logret_1 must equal log(close_t / close_{t-1}) recomputed from the panel
    panel = pd.read_parquet(PANEL_PATH)
    close = panel["Close"].unstack("ticker")
    expected = np.log(close / close.shift(1)).stack().reindex(features.index)
    assert np.allclose(features["logret_1"], expected, atol=1e-10)


@needs_data
def test_no_lookahead():
    """Truncating the future must not change any feature value in the past."""
    panel = pd.read_parquet(PANEL_PATH)
    full = feat.build_features(panel, cfg)
    dates = panel.index.get_level_values("date").unique().sort_values()
    cut = dates[len(dates) // 2]
    truncated = feat.build_features(panel.loc[panel.index.get_level_values("date") <= cut], cfg)
    common = truncated.index
    assert np.allclose(full.loc[common], truncated, atol=1e-12)


def test_graph_structure():
    edges = load_edge_list(cfg)
    g = build_graph(cfg)
    assert g.number_of_nodes() == 15
    assert g.number_of_edges() == len(edges)
    ei, ew = to_edge_index(g, cfg.universe.tickers, symmetrize=True)
    assert len(ei[0]) == len(ei[1]) == len(ew) == 2 * len(edges)
    # symmetrization is an exact mirror
    assert ei[0][len(edges):] == ei[1][: len(edges)]
    assert ei[1][len(edges):] == ei[0][: len(edges)]
