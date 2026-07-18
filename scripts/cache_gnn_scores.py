"""Phase 2.5 entrypoint: cache full-timeline GNN propagation scores."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.gnn.inference import run_inference

if __name__ == "__main__":
    scores = run_inference()
    print(f"cached {scores.shape[0]} days x {scores.shape[1]} pair scores "
          f"(mean {scores.values.mean():.3f}, std {scores.values.std():.3f})")
