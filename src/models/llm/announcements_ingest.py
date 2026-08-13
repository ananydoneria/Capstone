"""Phase 3.1: historical NSE corporate-announcements corpus, per (ticker, date).

Fetches via ``NseKit`` (``Nse.cm_live_hist_corporate_announcement``), which
wraps the same official nseindia.com endpoint with built-in cookie handling,
rate limiting, and retries. Raw records are cached per (symbol, month) under
``data/raw/announcements/`` — resume-safe, polite, and idempotent: existing
month files are never re-fetched.

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

from src.common.config import Config, load_config

REQUEST_GAP_S = 1.5  # polite spacing between API hits, on top of NseKit's own rate limit


def _corpus_dir(cfg: Config) -> Path:
    return Path(cfg.data.raw_dir) / "announcements"


def _month_starts(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"))


def fetch_all(cfg: Config | None = None) -> dict[str, int]:
    """Backfill month-by-month for every ticker over the RL window.

    Returns {symbol: fetched_month_count}; cached months are skipped.
    """
    from NseKit import Nse

    cfg = cfg or load_config()
    out_dir = _corpus_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    nse = Nse()
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
            try:
                df = nse.cm_live_hist_corporate_announcement(
                    symbol=symbol,
                    from_date=f"{month_start:%d-%m-%Y}",
                    to_date=f"{month_end:%d-%m-%Y}",
                )
                records = df.to_dict("records") if df is not None else []
                month_file.write_text(json.dumps(records, default=str), encoding="utf-8")
                fetched += 1
            except Exception as err:
                print(f"  WARN {symbol} {month_start:%Y-%m}: {type(err).__name__} — skipped")
            time.sleep(REQUEST_GAP_S)
        stats[symbol] = fetched
        print(f"{symbol}: {fetched} new month files")
    return stats


def parse_announcement_date(stamp: str) -> pd.Timestamp:
    """NSE timestamps look like '30-Jun-2026 18:29:34' (sometimes '30-06-2026...').
    Returns normalized date, or NaT if unparseable. NEVER slice [:10] — the
    '%d-%b-%Y' form is 11 chars and truncation corrupts the year."""
    head = str(stamp).strip().split(" ")[0]
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
        date = pd.to_datetime(head, format=fmt, errors="coerce")
        if pd.notna(date):
            return date.normalize()
    return pd.NaT


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
                date = parse_announcement_date(stamp)
                if text and pd.notna(date):
                    rows.append({"date": date, "ticker": ticker, "text": text})
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "text"])
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
