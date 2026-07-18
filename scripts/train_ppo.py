"""Phase 6 entrypoint: PPO training (compute-heavy; see src/rl_agent/train_ppo.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rl_agent.train_ppo import main

if __name__ == "__main__":
    main()
