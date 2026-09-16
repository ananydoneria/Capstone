# Capstone3 — Hybrid Neuro-Symbolic Trading System (NSE Auto Sector)

A four-layer algorithmic trading research system for the Indian Auto & Auto
Ancillaries sector, combining structural market awareness (a Graph Neural
Network over the sector's supply-chain graph) with semantic context (a local
Llama-3 sentiment engine), fused by a PPO reinforcement-learning portfolio
manager inside a cost-realistic simulated Indian exchange.

## Architecture

| Layer | Directory | Role |
|---|---|---|
| Data & Exchange Sandbox | `src/data_pipeline/`, `src/env/` | OHLCV ingestion (yfinance), supply-chain graph, Gymnasium exchange with Indian transaction costs |
| Structural (GNN) | `src/models/gnn/` | PyTorch Geometric model → shock-propagation confidence scores |
| Semantic (LLM) | `src/models/llm/` | Local quantized Llama-3-8B (Ollama, temp 0.0) → sentiment vectors in [-1, 1] |
| Execution (RL) | `src/rl_agent/` | Stable-Baselines3 PPO on pre-computed state vectors; Sharpe + drawdown reward |

Key design rule: **all GNN and LLM outputs are pre-computed for the full
historical timeline into `data/processed/state.h5`** — the RL training loop
never runs model inference.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pytest                            # Phase 0 gate
```

All tunables (universe, dates, seed, costs, hyperparameters) live in
[config.yaml](config.yaml). Data and library provenance: [Sources.md](Sources.md).

## Pipeline

```bash
python scripts/run_all.py            # cheap stages: data -> GNN -> state.h5 -> baselines -> report
python scripts/run_all.py --heavy    # + NSE news backfill, LLM sentiment, PPO (workstation)
```

Individual stages, in order:

| Stage | Command | Compute |
|---|---|---|
| 1. Data pipeline | `python scripts/run_phase1.py` | light (network, yfinance) |
| 2. GNN training | `python scripts/train_gnn.py` | light (CPU, ~1 min) |
| 2.5 GNN score cache | `python scripts/cache_gnn_scores.py` | light |
| 3a. News backfill | `python scripts/fetch_finnhub_news.py` + `fetch_google_news.py` | network-heavy |
| 3b. LLM sentiment | `python scripts/run_sentiment.py` | **workstation** — `ollama pull llama3:8b-instruct-q4_K_M` first |
| 4. State vector | `python scripts/build_state.py` | light |
| 5/6. Baselines | `python -m src.rl_agent.baselines` | light |
| 6a. PPO (×4 ablations) | `python scripts/train_ppo.py --ablation full\|no-gnn\|no-sentiment\|neither` | any machine — ~3 min per arm on CPU (measured 2026-09-16) |
| 6b. Evaluation | `python scripts/evaluate.py` | light |
| 7. Report | `python scripts/report.py` | light |

Without the LLM stage, `build_state.py` produces `state.h5` with a **neutral
sentiment placeholder** (flagged in file attrs); `train_ppo.py` warns if the
`full` / `no-gnn` arms are trained against it.
