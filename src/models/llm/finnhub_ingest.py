"""Phase 3.1b: supplementary financial news corpus via the Finnhub API.

Fetches ``/api/v1/company-news`` per (ticker, month) — the same cadence and
cache shape as ``announcements_ingest.py`` — under ``data/raw/finnhub/``.
Resume-safe: existing month files are never re-fetched.

Requires ``FINNHUB_API_KEY`` in the environment (see ``config.yaml``
``news.finnhub_api_key_env``); if unset, ``fetch_all``/``load_corpus`` return
empty results instead of raising — Finnhub is a supplementary source, never a
hard dependency (mirrors the neutral-sentiment-on-absence design).

Run:  python scripts/fetch_finnhub_news.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from src.common.config import Config, load_config

API = "https://finnhub.io/api/v1/company-news"
REQUEST_GAP_S = 1.0  # Finnhub free tier: 60 req/min


def _corpus_dir(cfg: Config) -> Path:
    return Path(cfg.data.raw_dir) / "finnhub"


def _api_key(cfg: Config) -> str | None:
    return os.environ.get(cfg.news.finnhub_api_key_env)


def _month_starts(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"))


def fetch_all(cfg: Config | None = None) -> dict[str, int]:
    """Backfill month-by-month for every ticker. Returns {symbol: fetched_count}."""
    cfg = cfg or load_config()
    if "finnhub" not in cfg.news.sources:
        print("finnhub not in news.sources — skipped")
        return {}
    key = _api_key(cfg)
    if not key:
        print(f"{cfg.news.finnhub_api_key_env} not set — skipping Finnhub backfill")
        return {}

    out_dir = _corpus_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    stats: dict[str, int] = {}

    for ticker in cfg.universe.tickers:
        sym_dir = out_dir / ticker.replace(".NS", "")
        sym_dir.mkdir(exist_ok=True)
        fetched = 0
        for month_start in _month_starts(cfg.data.start_date, cfg.data.end_date):
            month_file = sym_dir / f"{month_start:%Y-%m}.json"
            if month_file.exists():
                continue
            month_end = month_start + pd.offsets.MonthEnd(0)
            try:
                resp = sess.get(API, params={
                    "symbol": ticker,
                    "from": f"{month_start:%Y-%m-%d}",
                    "to": f"{month_end:%Y-%m-%d}",
                    "token": key,
                }, timeout=20)
                resp.raise_for_status()
                month_file.write_text(json.dumps(resp.json()), encoding="utf-8")
                fetched += 1
            except Exception as err:
                print(f"  WARN {ticker} {month_start:%Y-%m}: {type(err).__name__} — skipped")
            time.sleep(REQUEST_GAP_S)
        stats[ticker.replace(".NS", "")] = fetched
        print(f"{ticker}: {fetched} new month files")
    return stats


def load_corpus(cfg: Config | None = None) -> pd.DataFrame:
    """Flatten cached JSON into rows (date, ticker, text): headline + summary."""
    cfg = cfg or load_config()
    rows: list[dict] = []
    for ticker in cfg.universe.tickers:
        symbol = ticker.replace(".NS", "")
        for month_file in sorted((_corpus_dir(cfg) / symbol).glob("*.json")):
            try:
                items = json.loads(month_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            for item in items if isinstance(items, list) else []:
                text = " | ".join(
                    str(item[k]).strip() for k in ("headline", "summary") if item.get(k)
                )
                ts = item.get("datetime")
                date = pd.to_datetime(ts, unit="s", errors="coerce")
                if pd.notna(date):
                    date = date.normalize()
                if text and pd.notna(date):
                    rows.append({"date": date, "ticker": ticker, "text": text})
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "text"])
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
