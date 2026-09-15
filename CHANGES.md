# Changes Made — Summary & Justification

## Context
Root cause: nseindia.com blocks this sandbox at the Akamai bot-mitigation layer (connections silently dropped post-TLS-handshake, confirmed via `curl -v`). This isn't fixable from here, so per instruction ("ignore NSE entirely"), all NSE-direct sourcing was removed and replaced with verified working alternatives.

## Capstone repo changes (branch `match-deck-tech-stack`)

**OHLCV data source: nsepython/NseKit → yfinance**
- `src/data_pipeline/ohlcv_ingest.py` rewritten to pull from `yf.Ticker(ticker).history(...)` instead of nsepython.
- `src/rl_agent/paper_trade.py` updated to match (live-trading data fetch).
- Justification: yfinance hits Yahoo's own infrastructure, not nseindia.com — confirmed working live, and produced correct-shaped output (1237 days × 15 tickers, matching pre-branch baseline).

**News source: NSE announcements + Moneycontrol → Finnhub + Google News RSS**
- Deleted `src/models/llm/announcements_ingest.py` (NSE corporate announcements — blocked at the source).
- Deleted `src/models/llm/moneycontrol_ingest.py` (its autosuggest endpoint returns 404 — dead/stale, confirmed by direct test).
- Added `src/models/llm/google_news_ingest.py` — new module using Google News RSS (no API key, resume-safe month-by-month caching), covering both the corporate-announcements and Moneycontrol gaps.
- Kept `finnhub_ingest.py` unchanged — already worked, still a valid structured news source.
- Justification: Google News RSS was empirically verified to work with no auth and supports date-range queries, making it a drop-in replacement for two broken/blocked sources at once.

**Removed cross-validation**
- Deleted `src/data_pipeline/nse_crossval.py`.
- Justification: it existed to cross-check OHLCV against a second NSE feed; with NSE dropped entirely there's nothing left to cross-check against. Documented as a known limitation ("single-source OHLCV") rather than silently dropped.

**Config, dependencies, docs kept in sync**
- `config.yaml` / `src/common/config.py`: `news.sources` narrowed to `[finnhub, google_news]`.
- `requirements.txt`: removed `nsepython`, `nsekit`, `beautifulsoup4`, `lxml`; added `yfinance`.
- Scripts renamed/removed to match (`fetch_moneycontrol_news.py` → `fetch_google_news.py`; `fetch_announcements.py` deleted); `run_all.py` stage list updated.
- `Sources.md`, `README.md`, `HOWTORUN.txt` rewritten to describe the new pipeline instead of the old one.
- One test deleted (`test_nse_timestamp_parsing`) since the function it tested no longer exists.
- Justification: kept docs/tests/deps truthful to the actual code — a stale README or leftover dependency would mislead the next person.

**CLAUDE.md — two rounds of updates**
1. Documented the NSE-drop itself as a deliberate, accepted departure from the approved deck (explicitly flagged "not something to silently fix back without asking"), updated architecture description, DO-NOT rules, and STALE notes.
2. Added a **HANDOFF** section with copy-paste steps for moving the project to the stronger training PC: copy the whole folder (not `git clone`, since trained weights and cached data are git-ignored and only exist on disk), delete `.venv` (hardcodes absolute paths), reinstall, run `pytest` and expect `25 passed, 7 skipped`, then skip straight to STEP 1 (Ollama) — don't redo GNN training or data regen.
- Justification: CLAUDE.md is the project's single source of truth; anyone picking this up needs to know what changed, why, and exactly how to resume on different hardware without redoing already-finished work.

**Progress made on this sandbox before handoff**
- GNN trained and scored (val AUC 0.610; scores cached 1174 days × 64 pairs) — superseded by the GNN section at the end of this file.
- Google News corpus fetched in-repo (5,700 headlines, 2023-07 → 2026-06). Finnhub backfill skipped — no API key set here.
- Justification: front-loaded everything this sandbox *can* do, so the stronger PC only needs to pick up Ollama/PPO — explicitly deferred since this isn't the training machine.

## Standalone side-task (`news_data_pull/`, outside the git repo)

- New self-contained script `pull_news.py`, independent of Capstone's `src/`, pulling 5 years of Google News RSS headlines for the same 15-ticker universe.
- Result: 7,470 headlines, 2021-08-03 → 2026-08-17, per-ticker CSV output at `data/news_corpus.csv`.
- Sentiment analysis intentionally **not** implemented — explicitly deferred ("do later — just pull data for now").
- Justification: kept fully separate from the Capstone repo per stated scoping, so it doesn't entangle with project git history or dependencies.

## Everything explicitly deferred (not done, by instruction)
- Sentiment analysis on the standalone corpus.
- STEP 1–5 of the Capstone runbook (Ollama install, sentiment scoring, gating, PPO training) — reserved for the stronger PC.

## Commits on `match-deck-tech-stack`
1. `85d4593` — NSE-drop rewrite (yfinance + Finnhub/Google News swap, deletions, doc updates).
2. `f00a29e` — CLAUDE.md handoff instructions + progress log.

Nothing pushed to remote; nothing done beyond what was explicitly requested at each step.

## GNN data, label and evaluation (2026-09-11 → 2026-09-15)

**Corrected prices**
- `data/raw/corporate_actions.csv` + `ohlcv_ingest.apply_corporate_actions`, applied in `build_panel`: Yahoo adjusts splits but not demergers. TMPV 2025-10-14 (−51% → −1.1%) and MOTHERSON 2022-01-14 (+28% → −4.9%) were fake moves the simulated exchange would have booked as P&L. Raw downloads untouched; `ohlcv_panel`, `features`, GNN inputs and scores rebuilt. Guard test added.

**More data for the GNN**
- `src/data_pipeline/market_data.py` + `scripts/fetch_market_data.py`: 132 series back to 2010 (universe, Indian/Asian/Western indices, commodities, FX, rates, US autos, breadth baskets), leak-safe as-of joins (same day only for markets that close with or before India, else previous day), presence masking for SONACOMS before listing.
- GNN config gains `history_start`, `market_context`, `breadth`, `pair_features`, `ctx_pca`, `label_end_date`, `shock_basis`, `shock_vol_window`. Defaults reproduce the previous model exactly.
- `PropagationGNN` gains `graphconv` (uses the curated edge weights) and pair-head / context inputs. `paper_trade.todays_gnn` rebuilt on the same input builders.

**Evaluation**
- `scripts/gnn_walkforward.py`: expanding-window walk-forward on five half-year folds (2023H2–2025H2), 3 seeds, day-block bootstrap CIs, two no-learning rules on identical samples, volatility-stratified AUC, per-run checkpoints. 2026 never used.
- Finding: under the original label (raw return, 21d sigma) "lower previous-day volatility of the receiving stock" scored 0.64 AUC and beat every model. Cause: a 21d sigma mean-reverts inside the 5-day label window. Label redefined (sector-excess return, 63d sigma); the rule now scores 0.525.
- Result: V4 (2011+ history, GraphConv, pair features) 0.584 [0.535–0.636] vs previous shape 0.538, vol rule 0.525 (P>rule 91%), correlation 0.541. Market context never helped. Adopted by user decision; shipped weights retrained (val AUC 0.600).

**Reports and tests**
- `GNN_TEST_REPORT.md/.pdf`, `notebooks/gnn_testing.ipynb`, `GNN_NOTEBOOK_WALKTHROUGH.pdf` (new generator `scripts/gnn_notebook_walkthrough_pdf.py`) are fully data-driven; the volatility rule is always shown in its better direction.
- Tests 83 → 102 passed (8 skipped): corporate actions, as-of lag rules, presence masking, edge weights, PCA fit on train only, no-look-ahead for node/context/pair inputs, label never past `label_end_date`, excess/raw shock variants.
- `strongpc` preflight and `fix_06` include the new inputs; CLAUDE.md / HOWTORUN.txt / STATUS.md updated.
