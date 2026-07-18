"""Phase 8a entrypoint: daily paper-trading loop (no real orders)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rl_agent.paper_trade import main

if __name__ == "__main__":
    main()
