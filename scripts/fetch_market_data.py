"""Download the extended GNN dataset (universe history back to 2010, market
context, breadth baskets) and build the processed GNN inputs.

    python scripts/fetch_market_data.py            # uses cached downloads
    python scripts/fetch_market_data.py --refresh  # re-download everything

Resume-safe: each ticker is cached under data/raw/market/ on first download.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config
from src.data_pipeline import market_data as md

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    manifest = md.fetch_all(cfg, refresh=args.refresh)
    print(f"\nfetched {len(manifest['fetched'])} tickers, {len(manifest['failed'])} failed")
    for t, why in manifest["failed"].items():
        print(f"  FAILED {t}: {why}")
    summary = md.build_all(cfg)
    path = md.write_manifest(cfg, manifest, summary)
    print(f"\ncalendar {summary['calendar']}")
    print(f"context columns: {summary['context_columns']}  by group: {summary['context_groups']}")
    print(f"baskets: {summary['basket_sizes']}")
    print(f"manifest -> {path}")
