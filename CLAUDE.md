# CLAUDE.md — Capstone3 Project State & Roadmap (Claude's working notes)

> Auto-loaded each session. Update whenever state or plans change.

## What this project is

Hybrid neuro-symbolic algorithmic trading system for the NSE/BSE Indian Auto &
Auto Ancillaries sector. Four isolated layers (professional trading desk model):

1. **Data & Exchange Sandbox** (`src/data_pipeline/`, `src/env/`) — yfinance
   `.NS` OHLCV + curated supply-chain graph; custom Gymnasium env with Indian
   transaction costs, daily stepping.
2. **Structural** (`src/models/gnn/`) — PyG GNN over a NetworkX supply-chain
   graph → Propagation Confidence Scores (shock A cascades to B within k days).
3. **Semantic** (`src/models/llm/`) — local quantized Llama-3-8B-Instruct via
   Ollama/LangChain, temperature 0.0 → sentiment vector in [-1, +1].
4. **Execution** (`src/rl_agent/`) — SB3 PPO over a pre-computed state vector
   (portfolio + GNN scores + sentiment). Reward: differential Sharpe + drawdown
   penalty (cap 15%).

Hardware: local RTX 3090 (24GB). Separate from the Capstone2 CLOB simulator
(`C:\Users\shaha\Downloads\Capstone2`) — no shared code.

## Hard guardrails (do not regress)

1. **No LLM/GNN inference inside the RL loop.** Everything pre-computed into
   `data/processed/state.h5`; the env only samples from it.
2. **One seed in `config.yaml` (42)** → `src/common/seeding.py` covers
   random/NumPy/PyTorch; Gymnasium seeded at `reset(seed=)`, SB3 at `PPO(seed=)`.
   LLM temperature must be exactly 0.0 — config loader REJECTS anything else.
3. **Strict modularity**: layers never import each other; they communicate only
   through cached files. `src/common/` is the only shared import.
4. **Data safety**: only yfinance, jugaad-data, official NSE/BSE/SEBI endpoints.
   Every graph edge cites a source in Sources.md.
5. **No lookahead**: features at day t use data ≤ t; decisions at close(t) fill
   at open(t+1); GNN train/val split is temporal, never shuffled.

## User decisions (2026-07-19)

- Training window: **3 years** (2023-07-01 → 2026-06-30), walk-forward test
  from 2026-01-02.
- Reward: **Sharpe + drawdown penalty** (differential Sharpe, 15% DD cap).
- News corpus: **NSE/BSE corporate announcements** (official filings).
- Universe: 15 tickers (8 OEMs + 7 ancillaries) in config.yaml. Ticker fixes
  vs. the original brief: MOTHERSUMI→MOTHERSON.NS, BOSCHCHASS→BOSCHLTD.NS.

## Current state

Phase 0 DONE (scaffold, config.yaml + pydantic loader, seeding, Sources.md,
requirements.txt, .gitignore, tests/test_phase0.py — 4/4 passing in `.venv`
with numpy/pyyaml/pydantic/pytest installed; heavy deps not yet installed).

## Roadmap

- [x] Phase 0 — scaffold & reproducibility backbone
- [ ] Phase 1 — data pipeline: OHLCV ingest (yfinance, verify all 15 tickers
      resolve), jugaad-data cross-val, NSE-calendar alignment, features,
      relationships.csv + NetworkX graph builder
- [ ] Phase 2 — GNN: PyG dataset, shock/cascade labels, GraphSAGE model,
      temporal train, full-timeline inference cache, AUC vs correlation baseline
- [ ] Phase 3 — LLM: NSE/BSE announcements ingester, Ollama client (needs
      `ollama pull llama3:8b-instruct-q4_K_M` on user's box), batch sentiment
      cache, determinism check
- [ ] Phase 4 — state_builder → state.h5 + integrity gate (NaN/lookahead audit)
- [ ] Phase 5 — Gymnasium env: costs.py, portfolio.py, exchange_env.py,
      env_checker + deterministic replay test
- [ ] Phase 6 — PPO training + baselines (buy&hold, equal-weight, momentum)
      + ablations (no-GNN / no-LLM / neither)
- [ ] Phase 7 — results report, one-command repro script, finalize Sources.md

## Gotchas

- Windows venv: `.venv/Scripts/python` (not `bin/`). Old pip in venv lacks
  `--disable-pip-version-warning`.
- yfinance tickers with `&` (M&M.NS) and `-` (BAJAJ-AUTO.NS): quote in shells.
- torch must be installed with the cu121 index URL before torch-geometric
  (see requirements.txt comment).
- Multi-line git commit messages on PowerShell: write to file, `git commit -F`.
