# Capstone3 — Hybrid Neuro-Symbolic Trading System (NSE Auto Sector)

A four-layer algorithmic trading research system for the Indian Auto & Auto
Ancillaries sector, combining structural market awareness (a Graph Neural
Network over the sector's supply-chain graph) with semantic context (a local
Llama-3 sentiment engine), fused by a PPO reinforcement-learning portfolio
manager inside a cost-realistic simulated Indian exchange.

## Architecture

| Layer | Directory | Role |
|---|---|---|
| Data & Exchange Sandbox | `src/data_pipeline/`, `src/env/` | OHLCV ingestion, supply-chain graph, Gymnasium exchange with Indian transaction costs |
| Structural (GNN) | `src/models/gnn/` | PyTorch Geometric model → shock-propagation confidence scores |
| Semantic (LLM) | `src/models/llm/` | Local quantized Llama-3.2-3B (Ollama, temp 0.0) → sentiment vectors in [-1, 1] |
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
