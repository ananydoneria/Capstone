# Sources.md — Data, Libraries & Licensing Register

> Data-safety protocol: this project ingests data ONLY from the reputable
> endpoints listed here. No `.csv`/`.zip`/`.pkl` downloads from personal
> repositories or untrusted sites. Every supply-chain graph edge must carry a
> citation row in the "Relationship sources" table below.

## Data sources

| Source | Purpose | Access | License / Terms |
|---|---|---|---|
| Yahoo Finance via `yfinance` | Daily OHLCV bars for `.NS` tickers (auto-adjusted) | Python API | Apache-2.0 (library); Yahoo ToS for data — research/personal use |
| NSE bhavcopy via `jugaad-data` | Cross-validation of yfinance closes; official EOD prices | Python API | MIT (library); NSE data terms — personal/research use |
| NSE corporate announcements (nseindia.com) | Historical per-ticker filings text for LLM sentiment layer | Official NSE endpoint | NSE website terms — research use, rate-limited polite scraping |
| BSE corporate announcements (bseindia.com) | Supplementary filings coverage | Official BSE endpoint | BSE website terms — research use |
| SEBI filings / company annual reports | Supply-chain & related-party relationships for graph edges | Public disclosures | Public regulatory documents |

## Libraries

| Library | Layer | Purpose | License |
|---|---|---|---|
| `numpy`, `pandas`, `pyarrow` | all | Numerics, tabular data, Parquet IO | BSD-3 / Apache-2.0 |
| `yfinance` | data_pipeline | OHLCV ingestion | Apache-2.0 |
| `jugaad-data` | data_pipeline | NSE bhavcopy cross-validation | MIT |
| `networkx` | data_pipeline | Static supply-chain graph construction | BSD-3 |
| `torch` | gnn / rl | Deep learning runtime (CUDA, RTX 3090) | BSD-style |
| `torch-geometric` | gnn | GraphSAGE/GAT propagation-confidence model | MIT |
| `ollama` + Llama-3.2-3B-Instruct (q4_K_M) | llm | Local deterministic sentiment extraction (3B over 8B for faster batch pre-compute) | Ollama MIT; Llama 3.2 Community License |
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

- **Cross-validation (2026-07-19)**: yfinance closes matched official NSE
  bhavcopy (via `jugaad-data`) exactly — 21/21 overlapping days for MARUTI,
  max divergence 0.0000%. Tooling: `src/data_pipeline/nse_crossval.py`
  (tolerance 0.5%, since dividend auto-adjustment can shift closes slightly).
- **Tata Motors demerger (Oct 2025)**: `TATAMOTORS.NS` is retired on Yahoo.
  The universe uses `TMPV.NS` (Tata Motors Passenger Vehicles — the renamed
  original listed entity, full price history 2023-07 → present). `TMCV.NS`
  (CV spin-off) lists only from Dec 2025 and is excluded; note that
  CV-related supplier edges (e.g. Bharat Forge) pointed at the unified
  company for most of the window.
