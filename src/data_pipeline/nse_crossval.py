"""Cross-validate yfinance closes against official NSE bhavcopy (jugaad-data).

Best-effort guard against silent yfinance data corruption. NSE date columns
arrive as UTC-shifted timestamps (18:30 UTC == next-day midnight IST), hence
the +5h30m correction before normalizing.

Run:  python -m src.data_pipeline.nse_crossval  (network: nseindia.com)
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.common.config import Config, load_config
from src.data_pipeline.ohlcv_ingest import load_raw

TOLERANCE_PCT = 0.5  # auto_adjust dividend adjustments can shift closes slightly


def crossval_ticker(cfg: Config, ticker: str, days: int = 30) -> dict:
    from jugaad_data.nse import stock_df  # deferred: needs network

    end = pd.Timestamp(cfg.data.end_date).date()
    start = end - timedelta(days=days)
    nse = stock_df(symbol=ticker.replace(".NS", ""), from_date=start, to_date=end, series="EQ")
    d = (pd.to_datetime(nse["DATE"]) + pd.Timedelta(hours=5, minutes=30)).dt.normalize()
    nse_close = nse.assign(d=d).set_index("d")["CLOSE"].sort_index()

    yf_close = load_raw(cfg)[ticker]["Close"]
    both = pd.DataFrame({"nse": nse_close, "yf": yf_close}).dropna()
    div = ((both["nse"] - both["yf"]).abs() / both["nse"] * 100)
    return {"ticker": ticker, "days": len(both), "max_div_pct": float(div.max()) if len(both) else None}


def crossval(cfg: Config | None = None, sample: tuple[str, ...] = ("MARUTI.NS", "TMPV.NS", "BOSCHLTD.NS")) -> None:
    cfg = cfg or load_config()
    for ticker in sample:
        try:
            r = crossval_ticker(cfg, ticker)
            status = "OK" if (r["max_div_pct"] or 0) <= TOLERANCE_PCT else "DIVERGED"
            print(f"{status}  {ticker}: {r['days']} days, max divergence {r['max_div_pct']:.4f}%")
            if status == "DIVERGED":
                raise SystemExit(f"{ticker} diverges from NSE bhavcopy beyond {TOLERANCE_PCT}%")
        except SystemExit:
            raise
        except Exception as err:  # NSE endpoint is flaky — warn, don't block
            print(f"SKIP {ticker}: NSE endpoint unavailable ({type(err).__name__})")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    crossval()
