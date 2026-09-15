"""Extended GNN data: long price history for the universe, market context
(Indian sector indices, India VIX, Asian/Western indices, commodities, FX,
rates, global auto peers) and market-breadth baskets. All via yfinance.

Why: the GNN originally trained on 2021-07 onward only because SONACOMS listed
in June 2021. Every other universe name has history back to 2002-2007, so the
GNN panel here starts in 2011 (Nifty Auto index availability) and masks
SONACOMS as *absent* before it listed. Market context lets the model separate
"A's shock spread to B" from "the whole market moved".

Leakage rules (decision at NSE close on day t, 15:30 IST):
  * lag 0 — Indian indices and Asian markets (Tokyo/Seoul/Taipei/Shanghai/
    Hong Kong close before 15:30 IST): the value dated t is usable at t.
  * lag 1 — US/European indices, futures, FX, rates, US-listed autos: their
    session dated t closes AFTER the NSE close, so day t only sees values dated
    strictly before t (``merge_asof(allow_exact_matches=False)``).

Prices are split/bonus-adjusted by Yahoo; demergers are NOT, so
``data/raw/corporate_actions.csv`` back-adjusts them (e.g. the Tata Motors
PV/CV demerger on 2025-10-14, which otherwise shows as a fake -51% day).

Files (all under data/):
  raw/market/<SAFE>.parquet            adjusted Close/Volume per ticker
  processed/gnn_panel.parquet          universe Close/Volume/present, long
  processed/gnn_node_features.parquet  per-node features (NaN where absent)
  processed/market_context.parquet     [date x M] leak-safe context features
  processed/market_data_manifest.json  what was fetched, spans, failures

Run:  python scripts/fetch_market_data.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import Config, load_config
from src.data_pipeline.ohlcv_ingest import apply_corporate_actions, corporate_actions  # noqa: F401

HISTORY_DOWNLOAD_START = "2010-06-01"   # warmup ahead of gnn.history_start


@dataclass(frozen=True)
class Series:
    ticker: str
    name: str
    group: str        # india | asia | western | commodity | fx | global_auto
    lag: int          # 0 = usable same NSE day, 1 = previous session only
    kind: str = "price"  # price -> log returns; level -> level + differences


MACRO: list[Series] = [
    # Indian indices (same-day)
    Series("^NSEI", "nifty50", "india", 0), Series("^CNXAUTO", "nifty_auto", "india", 0),
    Series("^NSEBANK", "nifty_bank", "india", 0), Series("^CNXIT", "nifty_it", "india", 0),
    Series("^CNXMETAL", "nifty_metal", "india", 0), Series("^CNXENERGY", "nifty_energy", "india", 0),
    Series("^CNXFMCG", "nifty_fmcg", "india", 0), Series("^CNXPHARMA", "nifty_pharma", "india", 0),
    Series("^NSMIDCP", "nifty_next50", "india", 0), Series("^CNXINFRA", "nifty_infra", "india", 0),
    Series("^CNXREALTY", "nifty_realty", "india", 0), Series("^CNXPSUBANK", "nifty_psubank", "india", 0),
    Series("^INDIAVIX", "india_vix", "india", 0, "level"),
    # Asia (closes before the NSE close -> same day)
    Series("^N225", "nikkei", "asia", 0), Series("^HSI", "hang_seng", "asia", 0),
    Series("000001.SS", "shanghai", "asia", 0), Series("^KS11", "kospi", "asia", 0),
    Series("^TWII", "taiex", "asia", 0),
    # US / Europe (close after the NSE close -> lag 1)
    Series("^GSPC", "sp500", "western", 1), Series("^IXIC", "nasdaq", "western", 1),
    Series("^DJI", "dow", "western", 1), Series("^FTSE", "ftse", "western", 1),
    Series("^GDAXI", "dax", "western", 1), Series("^STOXX50E", "eurostoxx50", "western", 1),
    Series("^VIX", "us_vix", "western", 1, "level"), Series("^TNX", "us10y", "western", 1, "level"),
    Series("^IRX", "us3m", "western", 1, "level"),
    # Commodities (futures, US settlement -> lag 1)
    Series("CL=F", "wti", "commodity", 1), Series("BZ=F", "brent", "commodity", 1),
    Series("NG=F", "natgas", "commodity", 1), Series("GC=F", "gold", "commodity", 1),
    Series("SI=F", "silver", "commodity", 1), Series("HG=F", "copper", "commodity", 1),
    Series("PL=F", "platinum", "commodity", 1), Series("PA=F", "palladium", "commodity", 1),
    # FX (daily bar closes after NSE -> lag 1)
    Series("INR=X", "usdinr", "fx", 1), Series("DX-Y.NYB", "dxy", "fx", 1),
    Series("EURINR=X", "eurinr", "fx", 1), Series("JPYINR=X", "jpyinr", "fx", 1),
    # Global autos / industrials (US-listed -> lag 1)
    Series("TM", "toyota", "global_auto", 1), Series("HMC", "honda", "global_auto", 1),
    Series("F", "ford", "global_auto", 1), Series("GM", "gm", "global_auto", 1),
    Series("STLA", "stellantis", "global_auto", 1), Series("SLX", "steel_etf", "global_auto", 1),
    Series("XLI", "us_industrials", "global_auto", 1), Series("CARZ", "global_auto_etf", "global_auto", 1),
]

CONTEXT_GROUPS = {
    "none": set(),
    "india": {"india"},
    "full": {"india", "asia", "western", "commodity", "fx", "global_auto"},
}

# Breadth baskets. Large caps outside the universe (a Nifty-50-like basket,
# current membership — survivorship bias is harmless for a breadth signal) and
# auto/auto-ancillary names outside the universe (sector contagion breadth).
LARGECAP = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL",
    "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "SUNPHARMA", "TITAN", "ULTRACEMCO", "BAJFINANCE",
    "WIPRO", "NESTLEIND", "HCLTECH", "TECHM", "POWERGRID", "NTPC", "TATASTEEL", "JSWSTEEL",
    "ADANIENT", "ADANIPORTS", "ONGC", "COALINDIA", "BAJAJFINSV", "GRASIM", "HINDALCO", "DRREDDY",
    "CIPLA", "BRITANNIA", "APOLLOHOSP", "TATACONSUM", "SBILIFE", "HDFCLIFE", "INDUSINDBK", "BPCL",
    "SHRIRAMFIN", "TRENT", "BEL",
]
AUTO_EXT = [
    "MRF", "BALKRISIND", "TIINDIA", "CEATLTD", "ESCORTS", "ENDURANCE", "SUNDRMFAST", "SCHAEFFLER",
    "ZFCVINDIA", "OLECTRA", "FORCEMOT", "JAMNAAUTO", "GABRIEL", "SUPRAJIT", "LUMAXTECH", "MINDACORP",
    "CRAFTSMAN", "SANSERA", "VARROC", "ASAHIINDIA", "ARE&M", "JKTYRE", "SUBROS", "TIMKEN", "SKFINDIA",
    "WHEELS",
]
BASKETS = {"largecap": [f"{s}.NS" for s in LARGECAP], "auto_ext": [f"{s}.NS" for s in AUTO_EXT]}
MIN_BASKET_PRESENT = 5


# ------------------------------------------------------------------ download

def _safe(ticker: str) -> str:
    return (ticker.replace("^", "IDX_").replace("=", "_EQ_").replace("&", "_AND_")
            .replace(".", "_").replace("-", "_"))


def market_raw_dir(cfg: Config) -> Path:
    return Path(cfg.data.raw_dir) / "market"


def download(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """Adjusted daily bars (auto_adjust=True: splits+dividends) -> Open/Close/Volume."""
    import yfinance as yf

    last: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True, actions=False)
            if raw is not None and not raw.empty:
                out = raw[["Open", "Close", "Volume"]].copy()
                out.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
                out.index.name = "date"
                out = out[~out.index.duplicated(keep="last")].sort_index()
                return out[out["Close"] > 0]
        except Exception as err:
            last = err
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{ticker}: no data ({last})")


def fetch_all(cfg: Config | None = None, refresh: bool = False, verbose: bool = True) -> dict:
    """Download (or reuse cached) every ticker needed. Returns a manifest."""
    cfg = cfg or load_config()
    out_dir = market_raw_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    end = (pd.Timestamp(cfg.data.end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    wanted = ([(t, "universe") for t in cfg.universe.tickers]
              + [(s.ticker, f"macro:{s.group}") for s in MACRO]
              + [(t, f"basket:{b}") for b, ts in BASKETS.items() for t in ts])
    manifest = {"fetched": {}, "failed": {}}
    for i, (ticker, role) in enumerate(wanted, 1):
        path = out_dir / f"{_safe(ticker)}.parquet"
        try:
            if path.exists() and not refresh:
                df = pd.read_parquet(path)
            else:
                df = download(ticker, HISTORY_DOWNLOAD_START, end)
                df.to_parquet(path)
                time.sleep(0.3)
            manifest["fetched"][ticker] = {"role": role, "rows": int(len(df)),
                                           "start": str(df.index.min().date()),
                                           "end": str(df.index.max().date())}
            if verbose:
                print(f"[{i:3d}/{len(wanted)}] {ticker:16s} {role:18s} {len(df):5d} rows "
                      f"{df.index.min().date()} -> {df.index.max().date()}")
        except Exception as err:
            manifest["failed"][ticker] = f"{role}: {err}"
            if verbose:
                print(f"[{i:3d}/{len(wanted)}] {ticker:16s} {role:18s} FAILED: {err}")
    return manifest


def load_close(cfg: Config, ticker: str) -> pd.DataFrame | None:
    path = market_raw_dir(cfg) / f"{_safe(ticker)}.parquet"
    return pd.read_parquet(path) if path.exists() else None


# ------------------------------------------------------------------ corporate actions

# defined in ohlcv_ingest so the RL panel and the GNN history share one rule


# ------------------------------------------------------------------ builders (pure)

def shock_flags(ret1: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Same rule as labels.shock_matrix, on a wide return frame."""
    vol = ret1.rolling(cfg.features.volatility_window, min_periods=cfg.features.volatility_window).std()
    thr = (cfg.gnn.shock_z * vol.shift(1)).clip(lower=cfg.gnn.shock_min_move)
    return (ret1.abs() > thr) & ret1.notna()


def universe_frames(cfg: Config, frames: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    if frames is None:
        frames = {}
        for t in cfg.universe.tickers:
            df = load_close(cfg, t)
            if df is None:
                raise FileNotFoundError(f"{t} not downloaded — run scripts/fetch_market_data.py")
            frames[t] = df
    acts = corporate_actions(cfg)
    return {t: apply_corporate_actions(df, acts, t) for t, df in frames.items()}


def build_gnn_panel(cfg: Config, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Universe prices on the NSE calendar (union of trading dates on which at
    least half the universe traded). A ticker is *present* from its first bar
    on; gaps after listing (halts) are forward-filled with Volume 0."""
    counts = pd.Series(0, dtype=int)
    for df in frames.values():
        counts = counts.add(pd.Series(1, index=df.index), fill_value=0)
    calendar = pd.DatetimeIndex(sorted(counts[counts >= len(frames) / 2].index), name="date")
    rows = []
    for t, df in frames.items():
        d = df.reindex(calendar)
        listed = calendar >= df.index.min()
        d["Close"] = d["Close"].ffill().where(listed)
        d["Volume"] = d["Volume"].fillna(0.0).where(listed)
        d["present"] = listed.astype(float)
        d["ticker"] = t
        rows.append(d[["Close", "Volume", "present", "ticker"]].reset_index())
    return pd.concat(rows, ignore_index=True).set_index(["date", "ticker"]).sort_index()


def build_node_features(cfg: Config, panel: pd.DataFrame, ctx_index_close: pd.Series | None = None) -> pd.DataFrame:
    """Per-node features on the GNN calendar; NaN where absent or in warmup.
    Columns: the 6 legacy features, plus excess_ret_1 (vs Nifty Auto) when an
    index series is supplied."""
    f = cfg.features
    close = panel["Close"].unstack("ticker")
    volume = panel["Volume"].unstack("ticker")
    feats: dict[str, pd.DataFrame] = {}
    for w in f.return_windows:
        feats[f"logret_{w}"] = np.log(close / close.shift(w))
    feats[f"vol_{f.volatility_window}"] = feats["logret_1"].rolling(f.volatility_window).std()
    feats[f"mom_{f.momentum_window}"] = np.log(close / close.shift(f.momentum_window))
    vz = f.volume_zscore_window
    vmean, vstd = volume.rolling(vz).mean(), volume.rolling(vz).std()
    feats[f"volz_{vz}"] = ((volume - vmean) / vstd).replace([np.inf, -np.inf], np.nan)
    # zero-variance volume windows (halts) are neutral, but only where price exists
    feats[f"volz_{vz}"] = feats[f"volz_{vz}"].where(feats[f"volz_{vz}"].notna() | close.isna(), 0.0)
    if ctx_index_close is not None:
        idx = ctx_index_close.reindex(close.index).ffill()
        idx_ret = np.log(idx / idx.shift(1))
        feats["excess_ret_1"] = feats["logret_1"].sub(idx_ret, axis=0)
    long = pd.concat({k: v.stack(dropna=False) for k, v in feats.items()}, axis=1)
    long.index.names = ["date", "ticker"]
    return long.sort_index()


def _series_features(df: pd.DataFrame, s: Series, cfg: Config) -> pd.DataFrame:
    c = df["Close"]
    w = cfg.features.volatility_window
    if s.kind == "level":
        lvl = np.log(c) if s.name.endswith("vix") else c
        out = {f"{s.name}_level": lvl, f"{s.name}_chg1": lvl.diff(1), f"{s.name}_chg5": lvl.diff(5)}
    else:
        r = np.log(c / c.shift(1))
        out = {f"{s.name}_ret1": r, f"{s.name}_ret5": np.log(c / c.shift(5)),
               f"{s.name}_vol{w}": r.rolling(w).std()}
    return pd.DataFrame(out)


def _asof(feat: pd.DataFrame, calendar: pd.DatetimeIndex, lag: int, max_stale_days: int = 7) -> pd.DataFrame:
    """Value known at the NSE close of each calendar day, by the lag rule."""
    left = pd.DataFrame({"date": calendar})
    right = feat.dropna(how="all").reset_index().rename(columns={feat.index.name or "index": "src_date"})
    right = right.rename(columns={right.columns[0]: "src_date"})
    merged = pd.merge_asof(left, right, left_on="date", right_on="src_date",
                           allow_exact_matches=(lag == 0), direction="backward")
    stale = (merged["date"] - merged["src_date"]).dt.days > max_stale_days
    merged.loc[stale, feat.columns] = np.nan
    return merged.set_index("date")[list(feat.columns)]


def build_market_context(cfg: Config, calendar: pd.DatetimeIndex,
                         macro_frames: dict[str, pd.DataFrame],
                         basket_frames: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    parts = []
    for s in MACRO:
        df = macro_frames.get(s.ticker)
        if df is None or df.empty:
            continue
        feat = _series_features(df, s, cfg)
        feat.index.name = "src_date"
        aligned = _asof(feat, calendar, s.lag)
        aligned.columns = [f"{s.group}__{c}" for c in aligned.columns]
        parts.append(aligned)
    # breadth: same-day NSE stocks
    for bname, frames in basket_frames.items():
        closes = pd.DataFrame({t: df["Close"] for t, df in frames.items()}).reindex(calendar)
        listed = pd.DataFrame({t: calendar >= df.index.min() for t, df in frames.items()}, index=calendar)
        closes = closes.ffill().where(listed)
        ret = np.log(closes / closes.shift(1))
        n = ret.notna().sum(axis=1)
        ok = n >= MIN_BASKET_PRESENT
        shocks = shock_flags(ret, cfg)
        b = pd.DataFrame({
            f"breadth__{bname}_shock_frac": shocks.sum(axis=1) / n,
            f"breadth__{bname}_adv_frac": (ret > 0).sum(axis=1) / n,
            f"breadth__{bname}_mean_absret": ret.abs().mean(axis=1),
            f"breadth__{bname}_dispersion": ret.std(axis=1),
            f"breadth__{bname}_mean_ret": ret.mean(axis=1),
        }, index=calendar).where(ok)
        parts.append(b)
    ctx = pd.concat(parts, axis=1)
    ctx.index.name = "date"
    return ctx


def context_columns(cfg: Config, ctx: pd.DataFrame) -> list[str]:
    groups = CONTEXT_GROUPS[cfg.gnn.market_context]
    cols = [c for c in ctx.columns if c.split("__")[0] in groups]
    if cfg.gnn.breadth:
        cols += [c for c in ctx.columns if c.startswith("breadth__")]
    return cols


# ------------------------------------------------------------------ build + save

def processed(cfg: Config, name: str) -> Path:
    return Path(cfg.data.processed_dir) / name


def build_all(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    uni = universe_frames(cfg)
    panel = build_gnn_panel(cfg, uni)
    calendar = panel.index.get_level_values("date").unique()
    macro = {s.ticker: load_close(cfg, s.ticker) for s in MACRO}
    macro = {k: v for k, v in macro.items() if v is not None}
    baskets = {b: {t: df for t in ts if (df := load_close(cfg, t)) is not None} for b, ts in BASKETS.items()}
    auto_idx = macro.get("^CNXAUTO")
    nodes = build_node_features(cfg, panel, auto_idx["Close"] if auto_idx is not None else None)
    ctx = build_market_context(cfg, calendar, macro, baskets)
    panel.to_parquet(processed(cfg, "gnn_panel.parquet"))
    nodes.to_parquet(processed(cfg, "gnn_node_features.parquet"))
    ctx.to_parquet(processed(cfg, "market_context.parquet"))
    summary = {
        "calendar": [str(calendar.min().date()), str(calendar.max().date()), int(len(calendar))],
        "node_features": list(nodes.columns),
        "context_columns": int(ctx.shape[1]),
        "context_groups": {g: int(sum(c.startswith(g + "__") for c in ctx.columns))
                           for g in ["india", "asia", "western", "commodity", "fx", "global_auto", "breadth"]},
        "basket_sizes": {b: len(v) for b, v in baskets.items()},
        "macro_series": len(macro),
        "universe_first_bar": {t: str(df.index.min().date()) for t, df in uni.items()},
    }
    return summary


def live_inputs(cfg: Config, lookback_days: int = 450) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Fresh trailing window of GNN inputs for paper trading (no disk cache).
    Uses the identical builders as the offline pipeline. A context series that
    fails to download simply shows up as missing (neutral after scaling)."""
    end_ts = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = (end_ts - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = end_ts.strftime("%Y-%m-%d")
    uni = universe_frames(cfg, {t: download(t, start, end) for t in cfg.universe.tickers})
    panel = build_gnn_panel(cfg, uni)
    calendar = panel.index.get_level_values("date").unique()
    groups = CONTEXT_GROUPS[cfg.gnn.market_context]
    macro, baskets = {}, {}
    for s_ in MACRO:
        if s_.group in groups:
            try:
                macro[s_.ticker] = download(s_.ticker, start, end)
            except Exception as err:
                print(f"WARN live context {s_.ticker}: {err}")
    if cfg.gnn.breadth:
        for b, ts in BASKETS.items():
            baskets[b] = {}
            for t in ts:
                try:
                    baskets[b][t] = download(t, start, end)
                except Exception as err:
                    print(f"WARN live breadth {t}: {err}")
    idx = macro.get("^CNXAUTO")
    nodes = build_node_features(cfg, panel, idx["Close"] if idx is not None else None)
    ctx = build_market_context(cfg, calendar, macro, baskets) if (groups or cfg.gnn.breadth) else None
    return nodes, ctx


def load_node_features(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(processed(cfg, "gnn_node_features.parquet"))


def load_market_context(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(processed(cfg, "market_context.parquet"))


def write_manifest(cfg: Config, manifest: dict, summary: dict) -> Path:
    path = processed(cfg, "market_data_manifest.json")
    path.write_text(json.dumps({"download": manifest, "build": summary}, indent=2), encoding="utf-8")
    return path
