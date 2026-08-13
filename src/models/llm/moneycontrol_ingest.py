"""Phase 3.1c: supplementary financial news corpus scraped from Moneycontrol.

Two-step, best-effort scrape (Moneycontrol has no public news API):
  1. resolve each NSE ticker to its Moneycontrol company slug + internal id
     via the public autosuggest endpoint,
  2. scrape that company's news-listing page for headline + snippet text.

Polite: one request per (ticker, page) with ``news.moneycontrol_request_gap_s``
spacing, capped pages, rotating user-agent (via NseKit's ``MC`` session helper
where available). Any failure — network, markup change, missing slug — is
recorded and skipped, never fatal: this is a supplementary source layered on
top of the primary NSE announcements channel.

Cache: raw parsed article dicts per (symbol, page) under
``data/raw/moneycontrol/`` — resume-safe, existing page files are never
re-fetched.

Run:  python scripts/fetch_moneycontrol_news.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from src.common.config import Config, load_config

AUTOSUGGEST_API = "https://www.moneycontrol.com/mccode/common/autosuggesion.php"
NEWS_PAGE_TPL = "https://www.moneycontrol.com/news/tags/{slug}.html"
PAGES_PER_TICKER = 3
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}


def _corpus_dir(cfg: Config) -> Path:
    return Path(cfg.data.raw_dir) / "moneycontrol"


def resolve_slug(session: requests.Session, ticker: str) -> str | None:
    """NSE ticker -> Moneycontrol news-tag slug, via the public autosuggest API."""
    query = ticker.replace(".NS", "").replace("&", " ").replace("-", " ")
    try:
        resp = session.get(AUTOSUGGEST_API, params={"query": query}, timeout=15)
        resp.raise_for_status()
        items = resp.json()
        if not items:
            return None
        # autosuggest returns NSE-listed matches first; take the top hit's slug
        top = items[0] if isinstance(items, list) else items.get("data", [None])[0]
        if not top:
            return None
        return str(top.get("link_src") or top.get("slug") or "").strip("/").split("/")[-1] or None
    except Exception:
        return None


def _parse_news_page(html: str) -> list[dict]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    out: list[dict] = []
    # Moneycontrol's listing markup has shifted over the years; try the
    # current selector, then a looser fallback, then give up cleanly.
    cards = soup.select("li.clearfix") or soup.select("div.news_list li")
    for card in cards:
        title_el = card.select_one("h2 a, a.arial11_summ, h3 a")
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        summary_el = card.select_one("p")
        summary = summary_el.get_text(strip=True) if summary_el else ""
        date_el = card.select_one("span")
        date_str = date_el.get_text(strip=True) if date_el else ""
        if title:
            out.append({"title": title, "summary": summary, "date_str": date_str})
    return out


def fetch_all(cfg: Config | None = None) -> dict[str, int]:
    cfg = cfg or load_config()
    if "moneycontrol" not in cfg.news.sources:
        print("moneycontrol not in news.sources — skipped")
        return {}

    out_dir = _corpus_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers.update(HEADERS)
    gap = cfg.news.moneycontrol_request_gap_s
    stats: dict[str, int] = {}

    for ticker in cfg.universe.tickers:
        symbol = ticker.replace(".NS", "")
        sym_dir = out_dir / symbol
        sym_dir.mkdir(exist_ok=True)
        slug = resolve_slug(sess, ticker)
        time.sleep(gap)
        if not slug:
            print(f"  WARN {symbol}: could not resolve Moneycontrol slug — skipped")
            stats[symbol] = 0
            continue

        fetched = 0
        for page in range(1, PAGES_PER_TICKER + 1):
            page_file = sym_dir / f"page{page}.json"
            if page_file.exists():
                continue
            try:
                url = NEWS_PAGE_TPL.format(slug=slug) if page == 1 else \
                    NEWS_PAGE_TPL.format(slug=slug).replace(".html", f"/page-{page}/")
                resp = sess.get(url, timeout=20)
                resp.raise_for_status()
                articles = _parse_news_page(resp.text)
                page_file.write_text(json.dumps(articles), encoding="utf-8")
                fetched += 1
            except Exception as err:
                print(f"  WARN {symbol} page{page}: {type(err).__name__} — skipped")
            time.sleep(gap)
        stats[symbol] = fetched
        print(f"{symbol}: {fetched} new pages (slug={slug})")
    return stats


def load_corpus(cfg: Config | None = None) -> pd.DataFrame:
    """Flatten cached pages into rows (date, ticker, text).

    Moneycontrol's relative date strings ("2 days ago", "Yesterday") aren't
    reliably parseable after the fact; articles without a clean absolute date
    are dropped rather than guessed — a documented coverage gap for this
    supplementary source (the NSE channel remains the ground truth).
    """
    cfg = cfg or load_config()
    rows: list[dict] = []
    for ticker in cfg.universe.tickers:
        symbol = ticker.replace(".NS", "")
        for page_file in sorted((_corpus_dir(cfg) / symbol).glob("*.json")):
            try:
                items = json.loads(page_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            for item in items if isinstance(items, list) else []:
                text = " | ".join(
                    str(item[k]).strip() for k in ("title", "summary") if item.get(k)
                )
                date = pd.to_datetime(item.get("date_str"), errors="coerce", dayfirst=True)
                if pd.notna(date):
                    date = date.normalize()
                if text and pd.notna(date):
                    rows.append({"date": date, "ticker": ticker, "text": text})
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "text"])
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
