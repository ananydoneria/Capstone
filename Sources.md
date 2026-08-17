# Sources.md — Data, Libraries & Licensing Register

> Data-safety protocol: this project ingests data ONLY from the reputable
> endpoints listed here. No `.csv`/`.zip`/`.pkl` downloads from personal
> repositories or untrusted sites. Every supply-chain graph edge must carry a
> citation row in the "Relationship sources" table below.

## Data sources

**NSE-direct sourcing dropped (2026-08-17):** nseindia.com sits behind Akamai
bot-mitigation that silently drops non-residential/non-browser connections
post-TLS-handshake (not a clean 403) — unreliable for unattended runs. This
departs from the approved topic-approval deck's "NSEPython and NseKit"
tech-stack slide; a deliberate, documented tradeoff, not an oversight (see
CLAUDE.md).

| Source | Purpose | Access | License / Terms |
|---|---|---|---|
| Yahoo Finance via `yfinance` | Daily OHLCV bars for `.NS` tickers | Python wrapper, proxies Yahoo's own infrastructure | Apache-2.0 (library); Yahoo ToS — personal/research use |
| Finnhub company-news API | Primary financial news text for LLM sentiment layer | Official REST API, requires `FINNHUB_API_KEY` | Finnhub ToS — free tier, research/personal use |
| Google News RSS (`news.google.com/rss/search`) | Supplementary headline text for LLM sentiment layer | Public RSS feed, free-text + date-range query, no key | Google News ToS — research use |
| SEBI filings / company annual reports | Supply-chain & related-party relationships for graph edges | Public disclosures | Public regulatory documents |

## Libraries

| Library | Layer | Purpose | License |
|---|---|---|---|
| `numpy`, `pandas`, `pyarrow` | all | Numerics, tabular data, Parquet IO | BSD-3 / Apache-2.0 |
| `yfinance` | data_pipeline | OHLCV ingestion (Yahoo Finance) | Apache-2.0 |
| `networkx` | data_pipeline | Static supply-chain graph construction | BSD-3 |
| `torch` | gnn / rl | Deep learning runtime (CUDA, RTX 3090) | BSD-style |
| `torch-geometric` | gnn | GraphSAGE/GAT node encoder + temporal GRU propagation-confidence model | MIT |
| `ollama` + Llama-3-8B-Instruct (q4_K_M) | llm | Local deterministic sentiment extraction | Ollama MIT; Llama 3 Community License |
| `langchain`, `langchain-community` | llm | LLM orchestration, structured output | MIT |
| `pydantic` | common / llm | Config validation, LLM JSON schema enforcement | MIT |
| `h5py` | data_pipeline / env | Pre-computed state-vector cache (`state.h5`) | BSD-3 |
| `gymnasium` | env | Exchange sandbox environment API | MIT |
| `stable-baselines3` | rl_agent | PPO implementation | MIT |
| `tensorboard` | rl_agent | Training telemetry | Apache-2.0 |
| `pyyaml` | common | config.yaml parsing | MIT |
| `pytest` | tests | Test runner | MIT |

## Transaction-cost rate sources (config `env.costs`)

| Charge | Rate used | Source |
|---|---|---|
| STT (delivery, each side) | 0.1% | Union Budget 2024–25 / Income-tax provisions, as published by NSE |
| NSE transaction charge (capital mkt) | 0.00297% | NSE circulars on transaction charges |
| SEBI turnover fee | 0.0001% | SEBI circular on regulatory fees |
| Stamp duty (buy side, delivery) | 0.015% | Indian Stamp Act amendment (uniform stamp duty, 2020) |
| GST on statutory charges | 18% | CGST/SGST on brokerage & transaction charges |

*Rates to be re-verified against current NSE circulars before final results are reported.*

## Relationship sources (graph edges)

The full edge list lives in [`data/raw/relationships.csv`](data/raw/relationships.csv)
(32 directed supplier → buyer edges, one citation per row). Citations currently
reference the *class* of primary document (FY24 annual-report customer/segment
disclosures, DRHPs, investor presentations) and are tagged **VERIFY**: each must
be checked against the actual filing and the tag removed before final academic
submission.

## Data-integrity notes

- **Single-source OHLCV**: dropping NSE-direct sourcing means there is no
  longer an independent second feed to cross-validate `yfinance` closes
  against (the old `nse_crossval.py` cross-checked two NSE-official code
  paths against each other; both are gone). Documented limitation, not
  silently dropped.
- **Tata Motors demerger (Oct 2025)**: `TATAMOTORS.NS` is retired. The
  universe uses `TMPV.NS` (Tata Motors Passenger Vehicles — the renamed
  original listed entity, full price history 2023-07 → present). `TMCV.NS`
  (CV spin-off) lists only from Dec 2025 and is excluded; note that
  CV-related supplier edges (e.g. Bharat Forge) pointed at the unified
  company for most of the window.
- **Google News RSS**: `google_news_ingest.py` is rate-limited
  (`news.google_news_gap_s`) and best-effort — failures are recorded and
  skipped, never fatal. Headline-only text (no article body).
- **Finnhub coverage**: free-tier Finnhub's Indian-equity news coverage is
  thin/inconsistent; `finnhub_ingest.py` treats empty results as normal, not
  an error — Google News RSS supplements the gaps.
