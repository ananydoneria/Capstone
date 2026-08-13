"""Phase 3.1d: merge the NSE / Finnhub / Moneycontrol text corpora into one
(date, ticker, text) frame for the sentiment scorer.

Each source's ``load_corpus`` degrades gracefully on its own (missing cache,
missing API key, unresolved slug -> empty frame, never raises), so merging
here just needs to concatenate, dedup, and sort. Enabled sources are
controlled by ``config.yaml`` ``news.sources``.
"""

from __future__ import annotations

import pandas as pd

from src.common.config import Config, load_config
from src.models.llm import announcements_ingest, finnhub_ingest, moneycontrol_ingest

_LOADERS = {
    "nse": announcements_ingest.load_corpus,
    "finnhub": finnhub_ingest.load_corpus,
    "moneycontrol": moneycontrol_ingest.load_corpus,
}


def load_combined_corpus(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    frames = []
    for source in cfg.news.sources:
        df = _LOADERS[source](cfg)
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "text"])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "ticker", "text"])
    return combined.sort_values(["date", "ticker"]).reset_index(drop=True)
