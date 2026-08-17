"""Phase 3.1c: supplementary financial news corpus via Google News RSS.

Replaces the old NSE-corporate-announcements + Moneycontrol-scrape channels
(both dropped — see ``ohlcv_ingest.py`` docstring for why NSE-direct sourcing
was abandoned; Moneycontrol's autosuggest API returned 404 on the exact
endpoint used, i.e. dead/stale, and its scraper was never validated end to
end). Google News RSS needs no API key, no slug resolution, and no scraping —
it's a documented public feed (``news.google.com/rss/search``) that accepts
free-text queries plus ``after:``/``before:`` date operators.

Backfills month-by-month per ticker, same cache shape as the other ingesters:
raw parsed items under ``data/raw/google_news/<symbol>/<YYYY-MM>.json`` —
resume-safe, existing month files are never re-fetched.

Run:  python scripts/fetch_google_news.py
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

from src.common.config import Config, load_config

RSS_URL = "https://news.google.com/rss/search"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
}

# Query terms per ticker — plain NSE symbols are often too ambiguous/short
# for useful news search, so a few get a friendlier company-name hint.
_QUERY_HINTS = {
    "M&M.NS": "Mahindra Mahindra",
    "TMPV.NS": "Tata Motors",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "TVSMOTOR.NS": "TVS Motor",
    "EICHERMOT.NS": "Eicher Motors Royal Enfield",
    "ASHOKLEY.NS": "Ashok Leyland",
    "MOTHERSON.NS": "Motherson Sumi Samvardhana",
    "BOSCHLTD.NS": "Bosch India",
    "BHARATFORG.NS": "Bharat Forge",
    "SONACOMS.NS": "Sona Comstar",
    "UNOMINDA.NS": "UNO Minda",
    "EXIDEIND.NS": "Exide Industries",
    "APOLLOTYRE.NS": "Apollo Tyres",
}


def _query_term(ticker: str) -> str:
    return _QUERY_HINTS.get(ticker, ticker.replace(".NS", "")) + " NSE"


def _corpus_dir(cfg: Config) -> Path:
    return Path(cfg.data.raw_dir) / "google_news"


def _month_starts(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"))


def _parse_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out: list[dict] = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        if title and pub_date:
            out.append({"title": title, "pubDate": pub_date})
    return out


def fetch_all(cfg: Config | None = None) -> dict[str, int]:
    """Backfill month-by-month for every ticker. Returns {symbol: fetched_count}."""
    cfg = cfg or load_config()
    if "google_news" not in cfg.news.sources:
        print("google_news not in news.sources — skipped")
        return {}

    out_dir = _corpus_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers.update(HEADERS)
    gap = cfg.news.google_news_gap_s
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
            query = f"{_query_term(ticker)} after:{month_start:%Y-%m-%d} before:{month_end:%Y-%m-%d}"
            try:
                resp = sess.get(RSS_URL, params={"q": query, "hl": "en-IN", "gl": "IN"}, timeout=20)
                resp.raise_for_status()
                items = _parse_feed(resp.text)
                month_file.write_text(json.dumps(items), encoding="utf-8")
                fetched += 1
            except Exception as err:
                print(f"  WARN {symbol} {month_start:%Y-%m}: {type(err).__name__} — skipped")
            time.sleep(gap)
        stats[symbol] = fetched
        print(f"{symbol}: {fetched} new month files")
    return stats


def load_corpus(cfg: Config | None = None) -> pd.DataFrame:
    """Flatten cached month files into rows (date, ticker, text)."""
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
                title = str(item.get("title", "")).strip()
                try:
                    date = pd.Timestamp(parsedate_to_datetime(item["pubDate"])).tz_localize(None).normalize()
                except Exception:
                    continue
                if title:
                    rows.append({"date": date, "ticker": ticker, "text": title})
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "text"])
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def fetch_recent(ticker: str, cfg: Config | None = None, window: str = "1d") -> list[str]:
    """Live query for today's/recent headlines about one ticker (paper trading —
    no cache, no backfill, just the current feed)."""
    cfg = cfg or load_config()
    query = f"{_query_term(ticker)} when:{window}"
    try:
        resp = requests.get(RSS_URL, params={"q": query, "hl": "en-IN", "gl": "IN"},
                             headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return [it["title"] for it in _parse_feed(resp.text)]
    except Exception:
        return []
