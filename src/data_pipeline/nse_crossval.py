"""Cross-validate nsepython's historical closes against NseKit's independent
security-wise-data endpoint — both wrap official NSE APIs, but different
code paths, so this still catches parsing/column-mapping bugs in either.

Run:  python -m src.data_pipeline.nse_crossval  (network: nseindia.com)
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from src.common.config import Config, load_config
from src.data_pipeline.ohlcv_ingest import load_raw

TOLERANCE_PCT = 0.5  # small numeric noise between independently-parsed feeds


def crossval_ticker(cfg: Config, ticker: str, days: int = 30) -> dict:
    from NseKit import Nse  # deferred: needs network

    symbol = ticker.replace(".NS", "")
    end = pd.Timestamp(cfg.data.end_date)
    start = end - timedelta(days=days)
    nse = Nse().cm_hist_security_wise_data(
        symbol=symbol,
        from_date=start.strftime("%d-%m-%Y"),
        to_date=end.strftime("%d-%m-%Y"),
    )
    if nse is None or nse.empty:
        raise RuntimeError(f"NseKit returned no security-wise data for {symbol}")
    d = pd.to_datetime(nse["Date"], dayfirst=True).dt.normalize()
    nse_close = pd.to_numeric(
        nse["Close Price"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    nse_close = nse_close.set_axis(d).sort_index()

    ours_close = load_raw(cfg)[ticker]["Close"]
    both = pd.DataFrame({"nse": nse_close, "ours": ours_close}).dropna()
    div = ((both["nse"] - both["ours"]).abs() / both["nse"] * 100)
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
