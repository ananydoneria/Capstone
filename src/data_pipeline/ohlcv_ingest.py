"""Layer 1 ingestion: daily OHLCV bars for the configured universe via the
official NSE historical API, wrapped by ``nsepython`` (matches the topic
deck's tech-stack slide).

Writes one Parquet per ticker to ``data/raw/ohlcv/`` and a combined validated
panel to ``data/processed/ohlcv_panel.parquet`` (long format, one row per
(date, ticker), columns Open/High/Low/Close/Volume). NSE closes are
split-adjusted by the exchange itself; no dividend auto-adjustment is applied
(unlike the former yfinance source) — a documented limitation for the report.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.common.config import Config, load_config

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# The NSE historical-API JSON has used a few different field-naming schemes
# across endpoint revisions; try each candidate in order and fail loudly if
# none match (never silently fabricate a column).
_COL_CANDIDATES = {
    "date": ["CH_TIMESTAMP", "TIMESTAMP", "mTIMESTAMP", "Date"],
    "Open": ["CH_OPENING_PRICE", "OPEN", "Open Price"],
    "High": ["CH_TRADE_HIGH_PRICE", "HIGH", "High Price"],
    "Low": ["CH_TRADE_LOW_PRICE", "LOW", "Low Price"],
    "Close": ["CH_CLOSING_PRICE", "CLOSE", "Close Price"],
    "Volume": ["CH_TOT_TRADED_QTY", "VOLUME", "Total Traded Quantity"],
}


def effective_start(cfg: Config) -> str:
    """Download start: the GNN trains on extended history (gnn.train_start_date);
    the RL/backtest window (data.start_date) is a slice of it."""
    return min(cfg.data.start_date, cfg.gnn.train_start_date)


def _raw_path(cfg: Config, ticker: str) -> Path:
    safe = ticker.replace(".NS", "").replace("&", "_AND_").replace("-", "_")
    return Path(cfg.data.raw_dir) / "ohlcv" / f"{safe}.parquet"


def _pick_col(df: pd.DataFrame, field: str) -> str:
    for cand in _COL_CANDIDATES[field]:
        if cand in df.columns:
            return cand
    raise ValueError(
        f"nsepython response has no recognizable '{field}' column "
        f"(looked for {_COL_CANDIDATES[field]}; got {list(df.columns)})"
    )


def normalize_nse_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw NSE historical-API JSON -> our canonical OHLCV_COLS DataFrame,
    DatetimeIndex named 'date'."""
    date_col = _pick_col(raw, "date")
    out = pd.DataFrame(index=pd.to_datetime(raw[date_col], dayfirst=True).dt.normalize())
    out.index.name = "date"
    for field in OHLCV_COLS:
        col = _pick_col(raw, field)
        values = raw[col].astype(str).str.replace(",", "", regex=False)
        out[field] = pd.to_numeric(values, errors="coerce")
    return out.sort_index()


def download_ticker(cfg: Config, ticker: str, max_retries: int = 3) -> pd.DataFrame:
    """Fetch daily bars for one ticker over the configured window from the
    official NSE historical API."""
    from nsepython import equity_history

    symbol = ticker.replace(".NS", "")
    start = pd.Timestamp(effective_start(cfg)).strftime("%d-%m-%Y")
    end = pd.Timestamp(cfg.data.end_date).strftime("%d-%m-%Y")
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            raw = equity_history(symbol, "EQ", start, end)
            if raw is not None and not raw.empty:
                df = normalize_nse_history(raw)
                if not df.empty:
                    return df
        except Exception as err:  # network hiccups / NSE throttling: back off and retry
            last_err = err
        time.sleep(2**attempt)
    raise RuntimeError(f"No data for {ticker} after {max_retries} attempts: {last_err}")


def download_universe(cfg: Config | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    frames: dict[str, pd.DataFrame] = {}
    for ticker in cfg.universe.tickers:
        path = _raw_path(cfg, ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = download_ticker(cfg, ticker)
        df.to_parquet(path)
        frames[ticker] = df
    return frames


def load_raw(cfg: Config | None = None) -> dict[str, pd.DataFrame]:
    """Load previously downloaded per-ticker Parquets (no network)."""
    cfg = cfg or load_config()
    frames = {}
    for ticker in cfg.universe.tickers:
        path = _raw_path(cfg, ticker)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run download_universe() first")
        frames[ticker] = pd.read_parquet(path)
    return frames


def build_panel(frames: dict[str, pd.DataFrame], cfg: Config | None = None) -> pd.DataFrame:
    """Validate and align tickers onto the NSE trading calendar.

    Calendar = union of dates across tickers (all are liquid NSE names, so the
    union is the exchange calendar). Sparse gaps (halts) are forward-filled on
    prices with Volume=0; a ticker missing >2% of days fails loudly.
    """
    cfg = cfg or load_config()
    calendar = sorted(set().union(*(set(df.index) for df in frames.values())))
    calendar_idx = pd.DatetimeIndex(calendar, name="date")

    aligned = []
    report = {}
    for ticker, df in frames.items():
        missing = len(calendar_idx) - len(df.index.intersection(calendar_idx))
        report[ticker] = missing
        if missing / len(calendar_idx) > 0.02:
            raise ValueError(
                f"{ticker}: {missing}/{len(calendar_idx)} calendar days missing (>2%) — "
                "bad ticker or broken download, refusing to silently fill"
            )
        df = df.reindex(calendar_idx)
        df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].ffill()
        df["Volume"] = df["Volume"].fillna(0.0)
        df["ticker"] = ticker
        aligned.append(df.reset_index())

    panel = pd.concat(aligned, ignore_index=True).set_index(["date", "ticker"]).sort_index()
    if panel[["Open", "High", "Low", "Close"]].isna().any().any():
        # only possible if a ticker's series starts after the calendar start
        first_valid = {
            t: str(frames[t].index.min().date()) for t in frames if frames[t].index.min() > calendar_idx[0]
        }
        raise ValueError(f"Leading NaNs — tickers listing after window start: {first_valid}")

    panel.attrs["missing_report"] = report
    return panel


def save_panel(panel: pd.DataFrame, cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    out = Path(cfg.data.processed_dir) / "ohlcv_panel.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)
    return out
