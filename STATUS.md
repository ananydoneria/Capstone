# Project Status

Last updated: 2026-08-27, branch `match-deck-tech-stack`.

## What this is

Capstone trading system for the NSE Auto sector. Three parts: a GNN that
scores supply-chain shock propagation, a local LLM that scores news
sentiment, and a PPO agent that manages the portfolio in a simulated
exchange. They only talk to each other through cached files.

## Where things stand

Done:
- All code written and tested. 83 tests pass, 8 skip (7 skips need the
  sentiment step, which needs Ollama; 1 is an optional sklearn check —
  expected for now).
- GNN has its own test suite (`tests/gnn/`, 59 tests) and a report script
  (`scripts/gnn_test_report.py` → `GNN_TEST_REPORT.md`). Headline: val AUC
  0.610, beats correlation baseline (0.578) and vol baseline (0.352); seed
  range 0.610–0.629; relies mostly on `vol_21`.
- Data pulled: OHLCV via yfinance, 1237 days x 15 tickers.
- GNN trained (val AUC 0.610) and its scores cached.
- News corpus: 5,700 headlines from Google News RSS, covering the 3-year
  RL window (2023-07 to 2026-06). Finnhub is wired in too but skipped —
  no API key set on this machine.

Not done (needs the stronger PC, in this order):
1. Install Ollama, pull llama3:8b-instruct-q4_K_M.
2. Run sentiment scoring over the news corpus.
3. Rebuild state.h5 with real sentiment (must say `source: llm_cache`).
4. Train PPO — 4 ablation runs, 2M timesteps each. Hours.
5. Evaluate and write the report.

Note: nothing "trains the LLM". It's a pretrained model used as-is to
score sentiment. The GNN is already trained. PPO is the only training
left.

## The big change on this branch

Dropped NSE-direct data sourcing (nsepython/NseKit) entirely.
nseindia.com blocks non-browser traffic at the network level, verified by
testing. OHLCV now comes from yfinance, news from Finnhub + Google News
RSS. This deviates from the approved topic deck's tech-stack slide — done
knowingly, since unreachable sources beat matching the slide.

Side effect: the old cross-validation against a second NSE feed is gone,
so OHLCV is single-source now. Known limitation, noted in Sources.md.

## Moving to the stronger PC

Copy the whole folder — don't git clone, the data and trained weights are
git-ignored and only exist on disk. Delete `.venv`, recreate it, install
requirements, run the tests (expect 83 passed / 8 skipped), then start at
step 1 above. Full instructions in CLAUDE.md under HANDOFF.

If Claude Code isn't available there, the `strongpc/` folder has batch
scripts that do the same thing unattended: a preflight check with a score
and one fix script per missing item, a pipeline runner that logs every
stage and the GPU, and a collector that zips all reports. See
`strongpc/README.txt`.

## Separate side task

`../news_data_pull/` has a standalone script that pulled 5 years of
headlines (7,470, 2021-08 to 2026-08) for the same 15 tickers. Not part
of the main project. Sentiment analysis on it is planned but not started.

## Numbers to beat

On the H1-2026 test window (from the pre-branch pipeline, needs
regeneration before citing): momentum baseline +12.95%, Sharpe 1.02.
The thesis check is PPO-full beating PPO-neither.
