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
- [x] Phase 2.5 — gnn_scores.parquet cached (1174 days × 64 pairs, ran clean)
- [x] Phase 3 CODE — announcements_ingest (NSE cookie-handshake fetcher,
      month-cached, resume-safe), prompts (few-shot, calibrated), ollama_client
      (LangChain ChatOllama, injectable transport, None-on-any-failure),
      batch_sentiment (per-ticker checkpointing). **NOT RUN** — workstation only.
- [x] Phase 4 — state.h5 BUILT & verified (741 days; sentiment attr =
      neutral_placeholder until Phase 3 runs; rebuild after via build_state.py)
- [x] Phase 5 — env built & tested: costs.py (STT/txn/SEBI/stamp/GST/slippage,
      round-trip ~30bps), portfolio.py (sells-before-buys, cash-clamped, never
      negative), exchange_env.py (softmax weights incl. cash bucket, fill at
      open t+1, differential Sharpe + DD penalty, random 126d train episodes,
      ablation flags zero obs blocks at constant dims, bankruptcy at 10%).
      SB3 check_env passes; deterministic replay bit-identical.
- [x] Phase 6 CODE — train_ppo.py (4 ablation arms full/no-gnn/no-sentiment/
      neither, VecNormalize saved, warns on placeholder sentiment), evaluate.py
      (loads model+vecnorm, deterministic test replay, metrics+curve CSV).
      **PPO NOT TRAINED** (user away from GPU box). Baselines RUN on test:
      buy&hold -1.12%, equal-weight -1.51%, momentum +12.95% (Sharpe 1.02,
      maxDD 19.5%) — momentum is the bar PPO must beat.
- [x] Phase 7 CODE — run_all.py (cheap vs --heavy staging), report.py
      (graceful REPORT.md + equity_curves.png; smoke-tested, REPORT.md exists).
- [ ] WORKSTATION RUN (see runbook below)
- [ ] Verify relationships.csv citations against primary filings (VERIFY tags)

### Optional / post-capstone (user requested 2026-07-19 — "leave the door open")

- [ ] **Phase 8a — daily paper-trading loop.** Evening script (post-close):
      pull today's OHLCV bar + today's NSE announcements, compute the day's
      feature row, one GNN forward, LLM-score the filings, feed the trained
      PPO policy -> print target weights + the buy/sell order list for
      tomorrow's open. NO real orders. Append to a forward-test ledger
      (CSV: date, weights, hypothetical fills at next open, running P&L) so
      live effectiveness can be tracked manually against the backtest claim.
      Reuses every existing module; needs a small "incremental day" path in
      the pipeline (currently full-history batch) + policy/vecnorm loading.
      NOTE: the whole system is end-of-day by design — decisions at close t,
      fills at open t+1. Nothing runs during market hours.
- [ ] **Phase 8b — broker connectivity (Kotak Neo API), strictly optional.**
      Adapter layer only AFTER 8a has a convincing forward-test ledger:
      start with the Neo API's order-PLACEMENT sandbox/paper mode if
      available; real order execution is the user's own decision and
      responsibility, kept behind an explicit config flag defaulting off.
      Design seam already exists: Portfolio.rebalance produces the target
      trade list — a broker adapter would consume the same list.

## Workstation runbook (next session, in order)

1. `ollama pull llama3.2:3b-instruct-q4_K_M`
2. `python scripts/fetch_announcements.py` — NSE endpoint is flaky/rate-limited;
   resume-safe, re-run until no new month files appear. If NSE blocks entirely,
   fall back plan: BSE announcements or proceed with neutral sentiment and
   document the limitation.
3. `python scripts/run_sentiment.py` (resume-safe per ticker; ~3B model, fast)
4. `python scripts/build_state.py` → must print "source: llm_cache"
5. `python scripts/train_ppo.py --ablation full`, then `no-gnn`,
   `no-sentiment`, `neither` — or all heavy stages: `python scripts/run_all.py --heavy`
   (MlpPolicy PPO is small; CPU-only works too, just slower)
6. `python scripts/evaluate.py` then `python scripts/report.py`
7. Tests anytime: `.venv/Scripts/python -m pytest tests/ -q` (31 tests)

## Gotchas

- Windows venv: `.venv/Scripts/python` (not `bin/`). Old pip in venv lacks
  `--disable-pip-version-warning`.
- yfinance tickers with `&` (M&M.NS) and `-` (BAJAJ-AUTO.NS): quote in shells.
- torch: plain CPU wheel is installed and sufficient (tiny GNN + MlpPolicy);
  CUDA wheels are optional, only for faster PPO on the workstation.
- Baselines bypass the softmax action head by design (direct target weights
  through Portfolio.rebalance); PPO cannot express exact zero weights
  (softmax dust ~0.005% per asset) — economically irrelevant, but don't
  "fix" tests by expecting exact zeros.
- Multi-line git commit messages on PowerShell: write to file, `git commit -F`.
