"""Phase 7: one-command reproduction — raw data to final report.

    python scripts/run_all.py            # cheap stages only (no LLM, no PPO)
    python scripts/run_all.py --heavy    # everything (workstation: Ollama + hours of PPO)

Stage order (each idempotent; heavy stages are skipped without --heavy):
  1. run_phase1.py           download (yfinance) + validate + features + graph
  2. train_gnn.py            GNN training (CPU, ~1 min)
  3. cache_gnn_scores.py     full-timeline GNN inference cache
  4. fetch_finnhub_news.py     [heavy: needs FINNHUB_API_KEY]
  5. fetch_google_news.py      [heavy: network-intensive RSS backfill]
  6. run_sentiment.py          [heavy: needs Ollama serving the 8B model]
  7. build_state.py            fuse everything -> state.h5
  8. baselines                 cost-realistic baseline replays
  9. train_ppo.py x4 arms      [heavy: PPO, all ablations]
 10. evaluate.py               out-of-sample metrics per arm
 11. report.py                 REPORT.md + equity_curves.png
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

CHEAP = [
    ["scripts/run_phase1.py"],
    ["scripts/train_gnn.py"],
    ["scripts/cache_gnn_scores.py"],
]
HEAVY_PRE = [
    ["scripts/fetch_finnhub_news.py"],
    ["scripts/fetch_google_news.py"],
    ["scripts/run_sentiment.py"],
]
MID = [
    ["scripts/build_state.py"],
    ["-m", "src.rl_agent.baselines"],
]
HEAVY_TRAIN = [["scripts/train_ppo.py", "--ablation", arm]
               for arm in ("full", "no-gnn", "no-sentiment", "neither")]
POST = [
    ["scripts/evaluate.py"],
    ["scripts/report.py"],
]


def run(args: list[str]) -> None:
    print(f"\n=== {' '.join(args)} ===")
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def main() -> None:
    heavy = "--heavy" in sys.argv
    stages = CHEAP + (HEAVY_PRE if heavy else []) + MID
    if heavy:
        stages += HEAVY_TRAIN
    stages += POST
    if not heavy:
        print("cheap mode: skipping news backfill, LLM sentiment, and PPO training "
              "(pass --heavy on the workstation)")
    for stage in stages:
        run(stage)


if __name__ == "__main__":
    main()
