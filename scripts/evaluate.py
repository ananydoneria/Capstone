"""Phase 6 entrypoint: evaluate trained PPO arms out-of-sample."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rl_agent.evaluate import main

if __name__ == "__main__":
    main()
