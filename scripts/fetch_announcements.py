"""Phase 3 entrypoint A: backfill the NSE announcements corpus (network)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.llm.announcements_ingest import fetch_all, load_corpus

if __name__ == "__main__":
    fetch_all()
    corpus = load_corpus()
    print(f"corpus: {len(corpus)} announcements, "
          f"{corpus['ticker'].nunique() if len(corpus) else 0} tickers")
