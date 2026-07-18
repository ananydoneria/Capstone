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
- LLM: **llama3.2:3b-instruct-q4_K_M** (user chose smaller/faster over 8B).
- Universe: 15 tickers (8 OEMs + 7 ancillaries) in config.yaml. Ticker fixes
  vs. the original brief: MOTHERSUMI→MOTHERSON.NS, BOSCHCHASS→BOSCHLTD.NS,
  and **TATAMOTORS.NS→TMPV.NS** (Oct 2025 demerger; TMPV = renamed original
  entity with full history; TMCV.NS spin-off lists only Dec 2025, excluded).

## Current state

Phases 0–2 DONE (13/13 tests passing). `.venv` has pandas/yfinance/networkx/
jugaad-data/torch(CPU)/torch-geometric — CPU torch is fully sufficient for
this 15-node GNN; do NOT bother with CUDA wheels for it.

- Phase 1 artifacts (git-ignored, rebuild with `scripts/run_phase1.py`, or
  `--offline` to skip the download): ohlcv_panel.parquet (1237 days × 15
  tickers — GNN-only extended history from gnn.train_start_date 2021-07-01;
  RL window stays data.start_date 2023-07-01), features.parquet (1174 days
  × 15 × 6). Crossval vs NSE bhavcopy: exact match (0.0000%).
- relationships.csv: 32 directed edges, weakly connected, Bosch = top hub
  (degree 8). Citations are class-level, tagged VERIFY (see Sources.md).
- Phase 2 overfitting fix (user-requested): sample size 816→2715 via
  (a) 5y GNN history (earliest possible: SONACOMS IPO 2021-06-14),
  (b) volatility-scaled shocks |ret| > max(2.5σ_prev, 2%) instead of flat 4%,
  (c) bidirectional supervision (64 pairs). 2150 train / 565 val (17.7% /
  23.5% pos) over 302/77 shock days.
- GNN final (`scripts/train_gnn.py`): **val AUC 0.6314 @ epoch 3**, chosen by
  `scripts/sweep_gnn.py` (12-point grid; winner h64/d0.4/wd1e-3 seed-robust
  0.616–0.631 across seeds 42/7/2026). Train/val gap collapsed: train peaks
  0.73 (was 0.93). Baselines on identical val samples
  (`scripts/baseline_auc.py`): train-corr 0.585, dst-vol 0.359 (anti-
  predictive — z-scored shocks removed the vol prior). GNN beats structure-
  free priors by ~4.7 AUC pts; the decisive test remains the Phase 6 ablation.

## Roadmap

- [x] Phase 0 — scaffold & reproducibility backbone
- [x] Phase 1 — data pipeline: ingest, crossval, calendar alignment, features,
      relationships.csv + graph builder
- [x] Phase 2 (through training) — PyG dataset, shock/cascade labels,
      GraphSAGE model, temporal train w/ early stopping, weights saved;
      overfitting fixed (3.3x samples), sweep-selected config, baseline
      comparison (2.6) done
- [ ] Phase 2 remainder — full-timeline inference cache (2.5)
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
