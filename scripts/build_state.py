"""Phase 4 entrypoint: build data/processed/state.h5 from cached layers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline.state_builder import StateCache, build_state

if __name__ == "__main__":
    out = build_state()
    cache = StateCache()
    print(f"{out}: {len(cache)} days, features {cache.features.shape}, "
          f"gnn {cache.gnn.shape}, sentiment {cache.sentiment.shape} "
          f"(source: {cache.sentiment_source})")
