"""Phase 3 entrypoint B: batch sentiment pre-compute (GPU box, needs Ollama)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.llm.batch_sentiment import run_batch

if __name__ == "__main__":
    df = run_batch()
    if len(df):
        print(f"sentiment cache: {len(df)} (date, ticker) rows, "
              f"mean {df['sentiment'].mean():+.3f}")
