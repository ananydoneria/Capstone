"""Layer 1 ingestion: daily OHLCV bars for the configured universe via
``yfinance``. NSE-direct sourcing (nsepython/NseKit) was dropped — nseindia.com
sits behind Akamai bot-mitigation that blocks non-residential traffic outright
(silent TLS-handshake drop, not a clean 403), making it unreliable for
unattended/CI runs. Yahoo Finance proxies its own infrastructure and is not
subject to that block. This is a deliberate departure from the topic-approval
deck's "NSEPython and NseKit" tech-stack slide — documented, not silent.

Writes one Parquet per ticker to ``data/raw/ohlcv/`` and a combined validated
panel to ``data/processed/ohlcv_panel.parquet`` (long format, one row per
(date, ticker), columns Open/High/Low/Close/Volume). yfinance auto-adjusts for
splits and dividends by default; we request raw (unadjusted) OHLC so the
Indian cost model in ``src/env`` isn't fed already-adjusted prices.

Yahoo does NOT adjust demergers (e.g. Tata Motors -> TMPV + TMCV, ex-date
2025-10-14, shows as a -51% "crash"). Events listed in
``data/raw/corporate_actions.csv`` are back-adjusted in ``build_panel`` so the
simulated exchange doesn't book a loss that no shareholder suffered. The raw
per-ticker Parquets stay untouched.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.common.config import Config, load_config

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def effective_start(cfg: Config) -> str:
    """Download start: the GNN trains on extended history (gnn.train_start_date);
    the RL/backtest window (data.start_date) is a slice of it."""
    return min(cfg.data.start_date, cfg.gnn.train_start_date)


def _raw_path(cfg: Config, ticker: str) -> Path:
    safe = ticker.replace(".NS", "").replace("&", "_AND_").replace("-", "_")
    return Path(cfg.data.raw_dir) / "ohlcv" / f"{safe}.parquet"


def normalize_yf_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw yfinance ``Ticker.history()`` frame -> canonical OHLCV_COLS
    DataFrame, DatetimeIndex named 'date'."""
    out = raw[OHLCV_COLS].copy()
    out.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
    out.index.name = "date"
    return out.sort_index()


def download_ticker(cfg: Config, ticker: str, max_retries: int = 3) -> pd.DataFrame:
    """Fetch daily bars for one ticker over the configured window via yfinance."""
    import yfinance as yf

    start = pd.Timestamp(effective_start(cfg)).strftime("%Y-%m-%d")
    # yfinance's `end` is exclusive — push one day past to include end_date.
    end = (pd.Timestamp(cfg.data.end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            raw = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False, actions=False)
            if raw is not None and not raw.empty:
                df = normalize_yf_history(raw)
                if not df.empty:
                    return df
        except Exception as err:  # network hiccups / rate limiting: back off and retry
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


def corporate_actions(cfg: Config) -> pd.DataFrame:
    path = Path(cfg.data.raw_dir) / "corporate_actions.csv"
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "ex_date", "factor", "note"])
    return pd.read_csv(path, comment="#")


def apply_corporate_actions(df: pd.DataFrame, actions: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Back-adjust prices before each ex-date by the retained-value factor.

    factor 'auto' = Open(ex-date) / Close(previous day): the price at which the
    retained business was discovered, so the ex-date return reflects only
    trading on that day, not the value handed to shareholders as new shares.
    Volume is left alone (a demerger does not change the share count).
    """
    df = df.copy()
    price_cols = [c for c in ("Open", "High", "Low", "Close") if c in df.columns]
    for row in actions[actions["ticker"] == ticker].itertuples(index=False):
        ex = pd.Timestamp(row.ex_date)
        before = df.index < ex
        if not before.any() or ex not in df.index:
            continue
        if str(row.factor).strip().lower() == "auto":
            prev_close = df.loc[before, "Close"].iloc[-1]
            factor = float(df.loc[ex, "Open"]) / float(prev_close)
        else:
            factor = float(row.factor)
        df.loc[before, price_cols] *= factor
    return df


def build_panel(frames: dict[str, pd.DataFrame], cfg: Config | None = None,
                adjust: bool = True) -> pd.DataFrame:
    """Validate and align tickers onto the NSE trading calendar.

    Calendar = union of dates across tickers (all are liquid NSE names, so the
    union is the exchange calendar). Sparse gaps (halts) are forward-filled on
    prices with Volume=0; a ticker missing >2% of days fails loudly.
    ``adjust`` back-adjusts listed corporate actions (demergers) first.
    """
    cfg = cfg or load_config()
    if adjust:
        acts = corporate_actions(cfg)
        frames = {t: apply_corporate_actions(df, acts, t) for t, df in frames.items()}
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
