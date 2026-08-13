"""Phase 3.4: batch sentiment pre-compute over the merged news corpus
(NSE announcements + Finnhub + Moneycontrol, per ``news.sources``).

For every (trading day, ticker): score each article/filing, average the
successful scores into one daily sentiment value in [-1, 1]. Items on
non-trading days roll forward to the next trading day. Cache is written to
``data/processed/sentiment.parquet`` incrementally per ticker — resume-safe:
already-cached tickers are skipped on re-run.

Days with no corpus text are ABSENT from the cache; the Phase 4 state
builder fills them with cfg.llm.neutral_sentiment (0.0).

GPU-box only (needs Ollama serving the configured model):
    ollama pull llama3:8b-instruct-q4_K_M
    python scripts/run_sentiment.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import Config, load_config
from src.data_pipeline.features import load_features
from src.models.llm.news_corpus import load_combined_corpus
from src.models.llm.ollama_client import SentimentClient


def sentiment_path(cfg: Config) -> Path:
    return Path(cfg.data.processed_dir) / "sentiment.parquet"


def _align_to_trading_days(corpus: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Roll each announcement date forward to the first trading day >= it."""
    positions = calendar.searchsorted(corpus["date"])
    keep = positions < len(calendar)  # drop items after the last trading day
    corpus = corpus.loc[keep].copy()
    corpus["date"] = calendar[positions[keep]]
    return corpus


def run_batch(cfg: Config | None = None, client: SentimentClient | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    client = client or SentimentClient(cfg)

    calendar = pd.DatetimeIndex(
        load_features(cfg).index.get_level_values("date").unique()
    ).sort_values()
    corpus = _align_to_trading_days(load_combined_corpus(cfg), calendar)
    if corpus.empty:
        print("corpus is empty — run scripts/fetch_announcements.py "
              "(+ fetch_finnhub_news.py / fetch_moneycontrol_news.py) first")
        return pd.DataFrame(columns=["date", "ticker", "sentiment", "n_items", "n_failed"])

    path = sentiment_path(cfg)
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(
        columns=["date", "ticker", "sentiment", "n_items", "n_failed"]
    )
    done_tickers = set(cached["ticker"].unique())

    for ticker, group in corpus.groupby("ticker"):
        if ticker in done_tickers:
            print(f"{ticker}: cached, skipping")
            continue
        rows = []
        for date, day_group in group.groupby("date"):
            results = [client.score(text) for text in day_group["text"]]
            ok = [r.sentiment for r in results if r is not None]
            rows.append({
                "date": date,
                "ticker": ticker,
                "sentiment": sum(ok) / len(ok) if ok else cfg.llm.neutral_sentiment,
                "n_items": len(results),
                "n_failed": len(results) - len(ok),
            })
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cached.to_parquet(path)  # checkpoint after every ticker
        failed = sum(r["n_failed"] for r in rows)
        print(f"{ticker}: {len(rows)} trading days scored ({failed} item failures -> neutral)")

    return cached
