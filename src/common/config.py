"""Centralized, typed configuration loader.

Every module in the project obtains its parameters through ``load_config()``.
The pydantic models below make config.yaml self-validating: a typo'd key or
wrong type fails loudly at startup instead of silently corrupting a run.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ProjectCfg(BaseModel):
    name: str
    seed: int


class DataCfg(BaseModel):
    start_date: str
    end_date: str
    calendar: str
    raw_dir: str
    processed_dir: str
    cache_dir: str
    state_file: str
    relationships_file: str


class NewsCfg(BaseModel):
    sources: list[str]
    finnhub_api_key_env: str
    google_news_gap_s: float

    @field_validator("sources")
    @classmethod
    def _known_sources(cls, v: list[str]) -> list[str]:
        allowed = {"finnhub", "google_news"}
        if unknown := set(v) - allowed:
            raise ValueError(f"news.sources contains unknown entries: {unknown}")
        return v


class UniverseCfg(BaseModel):
    oems: list[str]
    ancillaries: list[str]

    @property
    def tickers(self) -> list[str]:
        return self.oems + self.ancillaries

    @field_validator("oems", "ancillaries")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("universe lists must not be empty")
        return v


class FeaturesCfg(BaseModel):
    return_windows: list[int]
    volatility_window: int
    momentum_window: int
    volume_zscore_window: int


class GnnCfg(BaseModel):
    arch: str
    hidden_dim: int
    num_layers: int
    dropout: float
    weight_decay: float
    train_start_date: str
    shock_z: float
    shock_min_move: float
    bidirectional_supervision: bool
    cascade_window_days: int
    temporal_window_days: int
    temporal_hidden_dim: int
    learning_rate: float
    epochs: int
    early_stopping_patience: int
    val_tail_fraction: float
    # Extended inputs (src/data_pipeline/market_data.py). Defaults reproduce
    # the original model exactly: 2021+ history, no context, no pair features.
    history_start: Optional[str] = None   # e.g. "2011-10-01"; None = legacy features.parquet
    market_context: str = "none"          # none | india | full
    breadth: bool = False                 # market-breadth basket features
    pair_features: List[str] = []         # subset of rolling_corr, edge_weight, direction
    corr_window: int = 63                 # trailing window for rolling_corr
    ctx_pca: int = 0                      # >0: compress market context to this many PCA factors (train-fit)
    label_end_date: Optional[str] = None  # no train/val label may read prices after this date
    # Shock definition. "raw": |logret_1| vs its own trailing sigma (legacy).
    # "excess": stock return minus Nifty Auto vs THAT series' sigma — strips
    # the common market move. shock_vol_window: sigma window for the threshold;
    # None = features.volatility_window (21). A 21d sigma mean-reverts fast, so
    # calm stocks clear it in the following week; 63d removes that shortcut.
    shock_basis: str = "raw"              # raw | excess
    shock_vol_window: Optional[int] = None

    @field_validator("arch")
    @classmethod
    def _known_arch(cls, v: str) -> str:
        if v not in {"graphsage", "gat", "graphconv"}:
            raise ValueError(f"gnn.arch must be graphsage | gat | graphconv, got {v!r}")
        return v

    @field_validator("shock_basis")
    @classmethod
    def _known_shock_basis(cls, v: str) -> str:
        if v not in {"raw", "excess"}:
            raise ValueError(f"gnn.shock_basis must be raw | excess, got {v!r}")
        return v

    @field_validator("market_context")
    @classmethod
    def _known_context(cls, v: str) -> str:
        if v not in {"none", "india", "full"}:
            raise ValueError(f"gnn.market_context must be none | india | full, got {v!r}")
        return v

    @field_validator("pair_features")
    @classmethod
    def _known_pair_features(cls, v: list[str]) -> list[str]:
        if unknown := set(v) - {"rolling_corr", "edge_weight", "direction"}:
            raise ValueError(f"gnn.pair_features contains unknown entries: {unknown}")
        return v


class LlmCfg(BaseModel):
    backend: str
    model: str
    temperature: float
    seed: int
    max_tokens: int
    neutral_sentiment: float

    @field_validator("temperature")
    @classmethod
    def _deterministic(cls, v: float) -> float:
        if v != 0.0:
            raise ValueError(
                "llm.temperature must be exactly 0.0 (deterministic reproducibility guardrail)"
            )
        return v


class CostsCfg(BaseModel):
    stt_delivery: float
    exchange_txn_nse: float
    sebi_turnover_fee: float
    stamp_duty_buy: float
    gst_on_charges: float
    slippage_bps: float


class RewardCfg(BaseModel):
    type: str
    sharpe_eta: float
    drawdown_cap: float
    drawdown_penalty_coef: float


class EnvCfg(BaseModel):
    initial_cash_inr: float
    fill_policy: str
    costs: CostsCfg
    reward: RewardCfg


class RlCfg(BaseModel):
    algo: str
    policy: str
    episode_length_days: int
    total_timesteps: int
    n_envs: int
    n_steps: int
    batch_size: int
    gamma: float
    gae_lambda: float
    learning_rate: float
    ent_coef: float
    train_end: str
    test_start: str


class Config(BaseModel):
    project: ProjectCfg
    data: DataCfg
    news: NewsCfg
    universe: UniverseCfg
    features: FeaturesCfg
    gnn: GnnCfg
    llm: LlmCfg
    env: EnvCfg
    rl: RlCfg

    model_config = {"frozen": True}


@functools.lru_cache(maxsize=4)
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate config.yaml. Cached: repeated calls are free."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)
