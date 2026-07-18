"""Phase 3.1: historical NSE corporate-announcements corpus, per (ticker, date).

Fetches from the official nseindia.com JSON endpoint (the site requires a
cookie handshake: hit the homepage first with browser-like headers). Raw JSON
is cached per (symbol, month) under ``data/raw/announcements/`` — resume-safe,
polite (rate-limited), and idempotent: existing month files are never re-fetched.

NSE availability note: the endpoint is rate-limited and occasionally blocks
non-browser traffic; failures are recorded and skipped, never fatal. Days with
no corpus simply yield neutral sentiment downstream.

Run:  python scripts/fetch_announcements.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from src.common.config import Config, load_config

NSE_BASE = "https://www.nseindia.com"
API = NSE_BASE + "/api/corporate-announcements"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NSE_BASE + "/companies-listing/corporate-filings-announcements",
}
REQUEST_GAP_S = 1.5  # polite spacing between API hits


def _corpus_dir(cfg: Config) -> Path:
    return Path(cfg.data.raw_dir) / "announcements"


def _month_starts(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"))


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    s.get(NSE_BASE, timeout=15)  # cookie handshake
    time.sleep(1.0)
    return s


def fetch_all(cfg: Config | None = None) -> dict[str, int]:
    """Backfill month-by-month for every ticker over the RL window.

    Returns {symbol: fetched_month_count}; cached months are skipped.
    """
    cfg = cfg or load_config()
    out_dir = _corpus_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = _session()
    stats: dict[str, int] = {}

    for ticker in cfg.universe.tickers:
        symbol = ticker.replace(".NS", "")
        sym_dir = out_dir / symbol
        sym_dir.mkdir(exist_ok=True)
        fetched = 0
        for month_start in _month_starts(cfg.data.start_date, cfg.data.end_date):
            month_file = sym_dir / f"{month_start:%Y-%m}.json"
            if month_file.exists():
                continue
            month_end = month_start + pd.offsets.MonthEnd(0)
            params = {
                "index": "equities",
                "symbol": symbol,
                "from_date": f"{month_start:%d-%m-%Y}",
                "to_date": f"{month_end:%d-%m-%Y}",
            }
            try:
                resp = sess.get(API, params=params, timeout=20)
                resp.raise_for_status()
                month_file.write_text(json.dumps(resp.json()), encoding="utf-8")
                fetched += 1
            except Exception as err:
                print(f"  WARN {symbol} {month_start:%Y-%m}: {type(err).__name__} — skipped")
                # refresh the cookie session before continuing
                try:
                    sess = _session()
                except Exception:
                    pass
            time.sleep(REQUEST_GAP_S)
        stats[symbol] = fetched
        print(f"{symbol}: {fetched} new month files")
    return stats


def load_corpus(cfg: Config | None = None) -> pd.DataFrame:
    """Flatten cached JSON into rows (date, ticker, text).

    Text = subject + description/attachment text where present. Announcement
    timestamps are mapped to their calendar date; alignment onto the trading
    calendar (weekends/holidays -> next trading day) happens in batch scoring.
    """
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
                    str(item[k]).strip()
                    for k in ("desc", "attchmntText", "sm_name", "subject")
                    if item.get(k) and str(item[k]).strip()
                )
                stamp = item.get("an_dt") or item.get("exchdisstime") or ""
                date = pd.to_datetime(str(stamp)[:10], errors="coerce")
                if text and date is not pd.NaT:
                    rows.append({"date": date.normalize(), "ticker": ticker, "text": text})
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "text"])
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
