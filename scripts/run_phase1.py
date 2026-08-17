"""Phase 1 entrypoint: download -> validate -> features -> graph. Idempotent.

Usage:  python scripts/run_phase1.py [--offline]
        --offline  skip the yfinance download, rebuild from data/raw parquets
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config
from src.common.seeding import set_global_seed
from src.data_pipeline import features as feat
from src.data_pipeline import ohlcv_ingest as ingest
from src.data_pipeline import relationships as rel


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg.project.seed)

    if "--offline" in sys.argv:
        frames = ingest.load_raw(cfg)
        print(f"[1/4] Loaded raw parquets for {len(frames)} tickers (offline)")
    else:
        frames = ingest.download_universe(cfg)
        print(f"[1/4] Downloaded {len(frames)} tickers "
              f"({ingest.effective_start(cfg)} .. {cfg.data.end_date})")

    panel = ingest.build_panel(frames, cfg)
    path = ingest.save_panel(panel, cfg)
    n_days = panel.index.get_level_values("date").nunique()
    print(f"[2/4] Panel: {n_days} trading days x {len(frames)} tickers -> {path}")
    print(f"      Missing-day report (gaps ffilled): "
          f"{ {k: v for k, v in panel.attrs['missing_report'].items() if v} or 'none'}")

    features = feat.build_features(panel, cfg)
    fpath = feat.save_features(features, cfg)
    fdays = features.index.get_level_values("date").nunique()
    print(f"[3/4] Features: {fdays} days x {len(cfg.universe.tickers)} tickers x "
          f"{features.shape[1]} features -> {fpath}")

    g = rel.build_graph(cfg)
    print(f"[4/4] Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} directed edges, "
          f"weakly connected: {__import__('networkx').is_weakly_connected(g)}")
    degrees = sorted(g.degree, key=lambda kv: -kv[1])[:3]
    print(f"      Highest-degree nodes: {degrees}")


if __name__ == "__main__":
    main()
