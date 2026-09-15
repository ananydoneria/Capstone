# CLAUDE.md — Capstone3 (auto-loaded; FOLLOW THE RUNBOOK BELOW EXACTLY)

Hybrid neuro-symbolic trading system, NSE Auto sector (academic capstone).
GNN supply-chain scores + local-LLM filing sentiment feed a PPO portfolio
manager inside a simulated Indian exchange. Everything is built and tested;
**the only remaining work is RUNNING the steps below** (LLM sentiment + PPO
training on the user's RTX 3090 PC), then evaluate + report.

**TO TRAIN THE MODEL: do STEPS 0→5 in order, then STEP 6 for results.**
The exact commands are written out in each step — run those files, nothing else.

ALWAYS use the venv's python, never bare `python`. Commands in this file are
written for Windows (`.venv\Scripts\python`); on Linux/macOS substitute
`.venv/bin/python` everywhere — same scripts, same order, nothing else
changes. Run everything from the project root (this folder). This file is the
source of truth; HOWTORUN.txt is the human-facing copy of the same runbook.

**What gets trained, so there is no confusion:**
- GNN — trainable, but ALREADY TRAINED (weights shipped in this folder). Do
  not retrain.
- LLM — NEVER trained. It is a pretrained Ollama model used inference-only
  to score sentiment (STEP 3).
- PPO — the ONLY thing left to train (STEP 5, 4 runs, hours, GPU helps).

---------------------------------------------------------------------------
## HANDOFF: moving this folder to the training PC (stronger machine)

This project was developed/data-prepped on a sandbox that is NOT the
training machine. GNN training and the news corpus backfill are already
done (see "Handoff progress" below) — do not redo them.

**No Claude Code on the training PC?** Use the self-driving toolkit in
`strongpc\` (read `strongpc\README.txt`): `00_preflight.bat` scores the
machine and names a `fix_XX_*.bat` per missing item; `10_run_pipeline.bat`
then runs STEPS 1→6 below unattended, logging every stage + GPU usage to
`reports\strongpc\run_<timestamp>\` and zipping the results. It enforces the
same gates and DO-NOT rules as this file. Manual route, on the new PC:

1. **Copy the whole `Capstone` folder** as-is (not a fresh `git clone` —
   `data/`, `src/models/gnn/weights/` are git-ignored and only exist on
   disk; a bare clone loses the already-trained GNN + news corpus).
2. **Delete the `.venv` folder** — it hard-codes absolute paths from this
   machine and will break. Everything else in the copy is fine untouched.
3. `python -m venv .venv` then
   `.venv\Scripts\python -m pip install -r requirements.txt`
4. `.venv\Scripts\python -m pytest tests -q` — expect `102 passed, 8 skipped`
   (7 skips need STEP 3 sentiment, which needs Ollama; 1 skip is an optional
   sklearn cross-check — both are correct at this point). If you see fewer
   passes or any failures, something didn't copy — STOP and investigate
   before continuing.
5. Skip straight to **STEP 1** below (Ollama). STEP 0's REGENERATE and the
   GNN sub-step are already done; do not rerun `run_phase1.py` /
   `train_gnn.py` / `cache_gnn_scores.py` unless you deliberately want fresh
   yfinance data (they're safe to rerun — idempotent — just unnecessary).
6. Optional: set `FINNHUB_API_KEY` in the environment and rerun
   `scripts/fetch_finnhub_news.py` before STEP 3 for a fuller news corpus
   (current corpus is Google-News-only, 5,700 headlines).

---------------------------------------------------------------------------
## STEP 0 — orient (every new session, do this first)

Run: `.venv\Scripts\python -m pytest tests -q`

| Result | Meaning | Action |
|---|---|---|
| `102 passed, 8 skipped` | EXPECTED state right now (7 skips need STEP 3 sentiment, which needs Ollama; 1 skip is an optional sklearn cross-check) | go to STEP 1 — do NOT run REGENERATE |
| `109 passed, 1 skipped` (or `110 passed`) | everything incl. sentiment present | check model.zip below; likely skip to STEP 6 |
| venv missing / import errors | fresh machine or copied folder | do SETUP below, retry |
| MORE than 8 skipped | data/weights didn't copy | do REGENERATE below, retry — but see WARNING |
| any failures | something broke | STOP; investigate, ask user before changing code |

WARNING: REGENERATE re-downloads fresh yfinance data and RETRAINS the GNN,
throwing away the already-trained weights and cached scores that shipped with
this folder. Only run it if more than 8 tests skip, and tell the user first.

**SETUP** (once): `python -m venv .venv` then
`.venv\Scripts\python -m pip install -r requirements.txt`
NOTE: if this folder was COPIED from another PC, the bundled `.venv` is
broken (venvs hard-code absolute paths). Any weird python/pip error →
delete the `.venv` folder entirely and do SETUP fresh. Everything else
(data/, weights, state.h5) copies fine and needs no regeneration.

**REGENERATE** (only if tests were skipped — ~10 min, needs internet):
```
.venv\Scripts\python scripts/run_phase1.py
.venv\Scripts\python scripts/fetch_market_data.py
.venv\Scripts\python scripts/train_gnn.py
.venv\Scripts\python scripts/cache_gnn_scores.py
.venv\Scripts\python scripts/build_state.py
```
(`fetch_market_data.py` pulls the 2011+ universe history, Nifty Auto and the
market-context series the GNN's label and inputs need; it caches per ticker
under `data/raw/market/` and is resume-safe.)

Also check: does `src/rl_agent/logs/ppo_full/model.zip` exist?
- YES → training already done. Do NOT retrain. Skip to STEP 6.
- NO → continue with STEP 1.

---------------------------------------------------------------------------
## STEP 1 — Ollama model (once per machine)

Run: `ollama list` — if `llama3:8b-instruct-q4_K_M` is missing:
`ollama pull llama3:8b-instruct-q4_K_M`
If ollama is not installed, tell the user to install it from https://ollama.com
(do not install it yourself), then continue from here.

## STEP 2 — news corpus backfill (network)

Run: `.venv\Scripts\python scripts/fetch_finnhub_news.py` then
`.venv\Scripts\python scripts/fetch_google_news.py`
- Both are resume-safe: re-run until every ticker reports `0 new month files`.
- Finnhub needs `FINNHUB_API_KEY` in the environment; skips cleanly if unset.
- If Google News RSS never returns anything: STOP and tell the user; proceed
  with neutral sentiment rather than improvising a new scraper.

## STEP 3 — LLM sentiment (needs Ollama running; resume-safe per ticker)

Run: `.venv\Scripts\python scripts/run_sentiment.py`
- Instant connection error → Ollama isn't serving; start it or ask the user.
- Item failures printing "-> neutral" are fine (by design, never crash).

## STEP 4 — rebuild state with real sentiment  **GATE**

Run: `.venv\Scripts\python scripts/build_state.py`
- MUST print `source: llm_cache`.
- If it prints `neutral_placeholder`: STEP 3 produced nothing. Go back.
  Do NOT proceed to STEP 5 past this gate without telling the user.

## STEP 5 — PPO training (the heavy part; 4 runs; hours total)

Run these one at a time, in this order, waiting for each to finish:
```
.venv\Scripts\python scripts/train_ppo.py --ablation full
.venv\Scripts\python scripts/train_ppo.py --ablation no-gnn
.venv\Scripts\python scripts/train_ppo.py --ablation no-sentiment
.venv\Scripts\python scripts/train_ppo.py --ablation neither
```
- Success per run = it prints `saved -> ...logs/ppo_<arm>` and model.zip
  exists there.
- 2,000,000 timesteps each (config.yaml `rl.total_timesteps`). Do NOT lower
  it to "save time" — partial runs are worthless for the ablation claim.
- If a run crashes: rerun that arm from scratch (it overwrites cleanly).
- Progress: `tensorboard --logdir src/rl_agent/logs` (optional).

## STEP 6 — evaluate + report (fast)

```
.venv\Scripts\python scripts/evaluate.py
.venv\Scripts\python scripts/report.py
```
Output: `REPORT.md` (metrics table: 4 PPO arms vs 3 baselines) and
`equity_curves.png` (PPO solid, baselines dashed). Tell the user the headline:
does `PPO full` beat `PPO neither` (the thesis) and beat momentum
(+12.95%, Sharpe 1.02 — the bar)? Then commit: REPORT.md, equity_curves.png,
and update the "Results so far" section of this file with the real numbers.

## STEP 7 — optional daily paper trading (only after STEP 5)

`.venv\Scripts\python scripts/paper_trade.py` once per trading day, evening
IST. Prints tomorrow's order list; NO real orders ever. Use `--no-llm` if
Ollama is unavailable. Ledger: `src/rl_agent/logs/paper/ledger.csv`.

---------------------------------------------------------------------------
## DO NOT (hard rules — no exceptions without explicit user approval)

- Do NOT edit `config.yaml`, the model architectures, the reward function,
  the universe/tickers, or the seed. Tuning is DONE; results must stay
  reproducible. (The `gnn:` block was last changed by explicit user decision
  on 2026-09-12/15 — see the dated comments inside it and "GNN update" below;
  that history is the model, not a licence to keep tuning.)
- Do NOT set LLM temperature to anything but 0.0 (the config loader rejects
  other values — that is intentional, do not "fix" it).
- Do NOT call the LLM or GNN inside the RL training loop. RL state comes
  ONLY from `data/processed/state.h5`.
- Do NOT retrain the GNN or PPO if their outputs already exist (GNN weights:
  `src/models/gnn/weights/propagation_gnn.pt`; PPO: `logs/ppo_*/model.zip`).
- Do NOT download datasets from anywhere except yfinance, Finnhub, or Google
  News RSS (already wired into the scripts). NSE-direct sourcing (nsepython/
  NseKit) was dropped — see "STALE" note below.
- Do NOT place, or wire up placing, real broker orders (Phase 8b is
  design-only; the user's own decision, later, separately).
- Do NOT lower total_timesteps, skip ablation arms, or estimate/fabricate
  results. Report actual numbers or report the failure.
- Do NOT commit `data/`, model weights, or `logs/` contents (gitignore
  handles it; the STEP 6 report files are the exception).

---------------------------------------------------------------------------
## Project knowledge (context; the runbook above takes precedence)

**Architecture — 4 isolated layers, communicating only via cached files:**
`src/data_pipeline/` (yfinance OHLCV, features, supply-chain graph, state.h5
builder) → `src/models/gnn/` (PyG GraphConv over the weighted supply-chain
graph + GRU temporal rollup + pair head → shock-cascade propagation scores) + `src/models/llm/` (Ollama
llama3:8b-instruct-q4_K_M, temp 0.0, Finnhub + Google News RSS →
sentiment in [-1,1]) → `src/env/` (Gymnasium exchange: Indian cost stack
~30bps round trip, decisions at close t fill at open t+1, differential
Sharpe reward + 15% drawdown penalty) → `src/rl_agent/` (SB3 PPO, 4 ablation
arms, baselines, evaluate, paper_trade).

**User decisions (2026-07-19):** 3-year RL window 2023-07-01→2026-06-30 with
test split from 2026-01-02; reward = Sharpe + drawdown penalty; LLM = 8B
(matches approved topic deck, supersedes the earlier 3B-for-speed tradeoff);
15 tickers fixed in config.yaml — note TATAMOTORS.NS→TMPV.NS (Oct-2025
demerger; TMCV.NS excluded, insufficient history), MOTHERSUMI→MOTHERSON.NS,
BOSCHCHASS→BOSCHLTD.NS.

**Deliberate deck departure (2026-08-17):** NSE-direct sourcing (nsepython/
NseKit) dropped entirely — nseindia.com sits behind Akamai bot-mitigation
that silently blocks non-residential/non-browser traffic, confirmed via live
testing (TLS handshake starts then hangs; `curl -v` shows it clearly). OHLCV
now via `yfinance`; news corpus now Finnhub + Google News RSS (`nse` and
`moneycontrol` sources removed, `nse_crossval.py` deleted — no second
NSE-official feed left to cross-check against). This is a known, accepted
mismatch against the approved topic deck's "NSEPython and NseKit" slide —
reachability won out over deck-literalism; not something to silently "fix"
back without asking first.

**STALE as of `match-deck-tech-stack` branch:** ingestion source (now
yfinance/Finnhub/Google-News, having gone via an intermediate nsepython/NseKit
attempt), LLM model (8B replacing 3B), and GNN architecture (temporal GRU
rollup replacing static per-day snapshot) all changed on this branch. Every
number below was measured on the original pre-branch pipeline and must be
fully regenerated (STEPS 0→6) before being cited — do not report these as
current results.

**Handoff progress (2026-08-17, done on a non-training sandbox — NOT the
RTX 3090 workstation, no Ollama installed here):**
- STEP 0 REGENERATE done: `ohlcv_panel.parquet` (yfinance, 1237 days × 15
  tickers), `features.parquet` built.
- STEP 0.5 (train_gnn.py + cache_gnn_scores.py) done — superseded by the
  "GNN update" block below (weights retrained 2026-09-15).
- STEP 2 partial: `fetch_google_news.py` done — 5,700 headlines, 15 tickers,
  2023-07-01→2026-06-30 cached under `data/raw/google_news/`.
  `fetch_finnhub_news.py` ran but skipped (no `FINNHUB_API_KEY` set) —
  corpus is Google-News-only until a key is provided.
- Tests: 102 passed, 8 skipped (7 need `build_state.py`, which needs Step 3
  sentiment — impossible without Ollama; 1 is an optional sklearn check).
- GNN test suite (`tests/gnn/`, 77 tests) + `scripts/gnn_test_report.py`
  (`GNN_TEST_REPORT.md`) + `scripts/gnn_test_pdf.py` (`GNN_TEST_REPORT.pdf`)
  + `scripts/gnn_notebook_walkthrough_pdf.py` (`GNN_NOTEBOOK_WALKTHROUGH.pdf`,
  explains `notebooks/gnn_testing.ipynb`). All numbers in them are computed
  live; run with `--seeds 7 2026` for seed retrains (~1 min each, temp dir,
  shipped weights untouched).

**GNN update (2026-09-11 → 15, user decisions; supersedes the GNN lines
above):**
- Data: two demergers Yahoo never adjusted (TMPV 2025-10-14 showed −51%,
  MOTHERSON 2022-01-14 +28%) are back-adjusted via
  `data/raw/corporate_actions.csv` in `ohlcv_ingest.build_panel`, so the RL
  panel, features and GNN inputs no longer contain fake moves.
  `scripts/fetch_market_data.py` adds the 2011+ universe history, Nifty Auto
  and market-context series (`data/raw/market/`, `data/processed/
  gnn_node_features.parquet`, `market_context.parquet`).
- Label: shock = |stock return − Nifty Auto return| > max(2.5 × previous-day
  63d sigma, 2%) (`gnn.shock_basis: excess`, `gnn.shock_vol_window: 63`).
  Under the old raw-return / 21d-sigma label a one-line "low previous-day
  volatility" rule scored 0.64 AUC out-of-sample and beat every model; under
  the new label it scores 0.525. No train/val label reads 2026 prices
  (`gnn.label_end_date: 2025-12-31`).
- Model: variant V4 — 2011+ history with presence masking, GraphConv (uses the
  curated edge weights), pair head with rolling 63d correlation + edge weight
  + direction. Chosen by `scripts/gnn_walkforward.py` (5 half-year folds
  2023H2–2025H2, 3 seeds, day-block bootstrap CIs):
  V4 0.584 (CI 0.535–0.636) vs previous shape 0.538, volatility rule 0.525
  (beaten in 91% of draws), correlation rule 0.541. V1/V3/V4 are within seed
  noise; market context (V5–V9) never helped. Report:
  `reports/gnn_walkforward/WALKFORWARD.md` (raw-label run archived beside it).
- Shipped weights retrained on that config: val AUC 0.600 (epoch 3), 4357
  train / 1206 val samples; `gnn_scores.parquet` recached (3635 days × 64).
  The modest margin is the honest result — the GNN is a small edge over
  simple rules, and the `no-gnn` PPO arm measures whether that edge matters.

**Handoff progress (continued):**
- **Not done, needs the real workstation:** STEP 1 (Ollama +
  llama3:8b-instruct-q4_K_M), STEP 3 (run_sentiment.py), STEP 4 gate
  (build_state.py), STEP 5 (PPO ×4 arms, hours). Copy this whole folder over
  (data/ + weights/ carry the above for free — do NOT rerun REGENERATE
  unless you want fresh yfinance data), delete `.venv`, redo SETUP, then
  pick up at STEP 1.
- Optional before STEP 1: set `FINNHUB_API_KEY` and rerun
  `fetch_finnhub_news.py` for a fuller news corpus.

**Results so far (pre-branch; all seeded with 42; 32/32 tests):**
- Data: 1237 days × 15 tickers (GNN trains on extended 2021-07 history; the
  RL window is the 3y slice). Sourced from yfinance; no independent
  cross-check feed remains (see "Deliberate deck departure" above).
- Graph: 32 directed supplier→buyer edges, Bosch top hub; citations tagged
  VERIFY in Sources.md (manual verification still in backlog).
- GNN: val AUC 0.6314 (sweep-selected h64/dropout0.4/wd1e-3; seed-robust
  0.616–0.631; beats correlation baseline 0.585 and vol baseline 0.359).
  Overfitting fixed via 2715 samples (5y history + z-scored shocks +
  bidirectional pairs); train/val gap collapsed (train peaks 0.73, was 0.93).
- state.h5: 741 RL days; sentiment currently `neutral_placeholder` (STEP 4
  replaces it). Env: SB3 check_env passes; replay bit-identical under seed.
- Baselines on the H1-2026 test window: buy&hold -1.12%, equal-weight
  -1.51%, momentum +12.95% (Sharpe 1.02, maxDD 19.5%) ← the bar PPO must beat.
- PPO: NOT trained yet — that is this session's job if model.zip is absent.

**Gotchas:**
- Windows: `.venv\Scripts\python`; multi-line git commit messages via a file
  and `git commit -F <file>`.
- CPU torch is sufficient everywhere (tiny GNN, small MlpPolicy); a CUDA
  wheel is optional and only speeds up PPO.
- paper_trade.py builds observations manually — its obs layout mirrors
  `exchange_env._obs`; if either changes, change both.
- PPO's softmax cannot emit exact-zero weights (~0.005% dust per asset) —
  expected and economically irrelevant; don't "fix" tests to demand zeros.
- Baselines intentionally bypass the softmax head (direct target weights
  through Portfolio.rebalance).

**Backlog (not this session):** verify relationships.csv citations against
primary filings; Phase 8b Kotak Neo broker adapter (design-only, gated on a
convincing paper-trade ledger + explicit user decision); optional Streamlit
dashboard (user undecided).
