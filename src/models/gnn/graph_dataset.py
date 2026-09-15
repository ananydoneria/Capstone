"""Bridge from pipeline artifacts to PyTorch tensors for the propagation GNN.

Two node-feature sources:
  * legacy  (gnn.history_start is None): data/processed/features.parquet —
    2021+ history, every node present every day. Reproduces the original model.
  * history (gnn.history_start set): data/processed/gnn_node_features.parquet
    from market_data.py — history back to 2011, split/demerger-adjusted
    prices, nodes *absent* before they listed (SONACOMS pre mid-2021). Absent
    node-days get zero features plus a 0 in an extra ``present`` channel, never
    generate training samples, and are excluded from normalisation statistics.

Optional extras, both leak-safe by construction (see market_data.py):
  * market context [T, M] fed to the pair head (config gnn.market_context /
    gnn.breadth), z-scored with train-period statistics;
  * pair features [T, P, K] (gnn.pair_features): trailing rolling correlation
    of the two stocks' returns (data <= t), curated link weight, direction.

Everything is small (15 nodes, a few thousand days), so it lives in RAM as
dense tensors. Normalisation uses TRAIN-period statistics only; the stats are
persisted with the weights so inference and paper trading match exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from src.common.config import Config, load_config
from src.data_pipeline.relationships import build_graph, to_edge_index
from src.models.gnn.labels import CascadeSample, build_samples, shock_matrix, temporal_split

CORE_FEATURES = ["logret_1", "logret_5", "logret_21", "vol_21", "mom_63", "volz_21"]
CTX_CLIP = 8.0


@dataclass
class GraphData:
    tickers: list[str]
    dates: list[pd.Timestamp]
    x: torch.Tensor            # [T, N, F] normalised node features
    edge_index: torch.Tensor   # [2, E] symmetrised message-passing edges
    edge_weight: torch.Tensor  # [E]
    directed_pairs: list[tuple[int, int]]   # supervision pairs as node indices (scoring order)
    train: list[CascadeSample]
    val: list[CascadeSample]
    norm_mean: torch.Tensor    # [F_core] train-period stats (persisted with weights)
    norm_std: torch.Tensor
    pair_feat: torch.Tensor | None = None   # [T, P, K]
    ctx: torch.Tensor | None = None         # [T, M]
    ctx_mean: torch.Tensor | None = None
    ctx_std: torch.Tensor | None = None
    present: torch.Tensor | None = None     # [T, N] bool
    ret1: pd.DataFrame | None = None        # raw daily log returns (dates x tickers)
    vol: pd.DataFrame | None = None         # raw trailing volatility (dates x tickers)
    node_feature_names: list[str] = field(default_factory=list)
    ctx_names: list[str] = field(default_factory=list)
    pair_feature_names: list[str] = field(default_factory=list)
    extra_stats: dict | None = None          # e.g. PCA basis, persisted with the weights

    @property
    def pair_dim(self) -> int:
        return 0 if self.pair_feat is None else int(self.pair_feat.shape[-1])

    @property
    def ctx_dim(self) -> int:
        return 0 if self.ctx is None else int(self.ctx.shape[-1])

    def day_index(self, date: pd.Timestamp) -> int:
        return self._date_pos[date]

    def pair_id(self, src: int, dst: int) -> int:
        return self._pair_pos[(src, dst)]

    def window(self, day_idx: int, width: int) -> torch.Tensor:
        """Trailing window of ``width`` days ending at (inclusive) day_idx,
        oldest -> newest. Left-zero-padded near the start of the timeline —
        every real day used is still <= day_idx, so no lookahead leaks in."""
        return window_features(self.x, day_idx, width)

    def __post_init__(self) -> None:
        self._date_pos = {d: i for i, d in enumerate(self.dates)}
        self._pair_pos = {tuple(p): i for i, p in enumerate(self.directed_pairs)}


def features_tensor(
    features: pd.DataFrame, tickers: list[str]
) -> tuple[list[pd.Timestamp], torch.Tensor]:
    """Long-format features -> (dates, [T, N, F] tensor), ticker order enforced."""
    dates = list(features.index.get_level_values("date").unique())
    wide = features.unstack("ticker")  # columns: (feature, ticker)
    blocks = [
        torch.tensor(wide[name][tickers].to_numpy(), dtype=torch.float32)
        for name in features.columns
    ]
    return dates, torch.stack(blocks, dim=-1)


def window_features(x: torch.Tensor, day_idx: int, width: int) -> torch.Tensor:
    """Standalone version of ``GraphData.window`` for callers holding a bare
    [T, N, F] tensor (e.g. inference/paper-trading) rather than a GraphData."""
    lo = day_idx - width + 1
    if lo >= 0:
        return x[lo : day_idx + 1]
    pad = torch.zeros((-lo, *x.shape[1:]), dtype=x.dtype)
    return torch.cat([pad, x[: day_idx + 1]], dim=0)


def supervision_pairs(cfg: Config) -> list[tuple[str, str]]:
    """Directed (src, dst) ticker pairs the edge head scores, config-aware."""
    g = build_graph(cfg)
    pairs = [(u, v) for u, v in g.edges()]
    if cfg.gnn.bidirectional_supervision:
        pairs += [(v, u) for u, v in g.edges()]
    return pairs


# ------------------------------------------------------------------ raw inputs

def node_frame(cfg: Config) -> pd.DataFrame:
    """Long (date, ticker) node features per the configured source."""
    if cfg.gnn.history_start is None:
        from src.data_pipeline.features import load_features
        return load_features(cfg)
    from src.data_pipeline.market_data import load_node_features
    nodes = load_node_features(cfg)
    return nodes.loc[nodes.index.get_level_values("date") >= pd.Timestamp(cfg.gnn.history_start)]


def context_frame(cfg: Config) -> pd.DataFrame | None:
    if cfg.gnn.market_context == "none" and not cfg.gnn.breadth:
        return None
    from src.data_pipeline.market_data import load_market_context
    return load_market_context(cfg)


def _node_columns(cfg: Config, nodes: pd.DataFrame) -> list[str]:
    cols = [c for c in CORE_FEATURES if c in nodes.columns]
    if cfg.gnn.market_context != "none" and "excess_ret_1" in nodes.columns:
        cols.append("excess_ret_1")
    return cols


def _pair_features(cfg: Config, ret1: pd.DataFrame, pair_names: list[tuple[str, str]]) -> torch.Tensor | None:
    wanted = cfg.gnn.pair_features
    if not wanted:
        return None
    g = build_graph(cfg)
    w = cfg.gnn.corr_window
    blocks = []
    for u, v in pair_names:
        cols = []
        if "rolling_corr" in wanted:
            corr = ret1[u].rolling(w, min_periods=w // 2).corr(ret1[v])
            cols.append(corr.fillna(0.0).clip(-1, 1).to_numpy())
        if "edge_weight" in wanted:
            weight = g.edges[u, v]["weight"] if g.has_edge(u, v) else g.edges[v, u]["weight"]
            cols.append(np.full(len(ret1), float(weight)))
        if "direction" in wanted:
            cols.append(np.full(len(ret1), 1.0 if g.has_edge(u, v) else 0.0))
        blocks.append(np.stack(cols, axis=-1))
    return torch.tensor(np.stack(blocks, axis=1), dtype=torch.float32)  # [T, P, K]


def raw_inputs(cfg: Config, nodes: pd.DataFrame | None = None,
               ctx_df: pd.DataFrame | None = None, ctx_names: list[str] | None = None) -> dict:
    """Unnormalised tensors + frames for the whole node-feature calendar."""
    cfg = cfg or load_config()
    tickers = cfg.universe.tickers
    nodes_full = node_frame(cfg) if nodes is None else nodes
    cols = _node_columns(cfg, nodes_full)
    nodes = nodes_full[cols]                       # model inputs; labels see the full frame
    dates, x_raw = features_tensor(nodes, tickers)
    # a node-day is present when its core features are all defined
    present = ~torch.isnan(x_raw[..., : len([c for c in cols if c in CORE_FEATURES])]).any(-1)

    g = build_graph(cfg)
    ei, ew = to_edge_index(g, tickers, symmetrize=True)
    pos = {t: i for i, t in enumerate(tickers)}
    pair_names = supervision_pairs(cfg)
    ret1 = nodes["logret_1"].unstack("ticker")[tickers]
    vol_col = f"vol_{cfg.features.volatility_window}"
    vol = nodes[vol_col].unstack("ticker")[tickers]

    ctx_raw, ctx_names = None, []
    if cfg.gnn.market_context != "none" or cfg.gnn.breadth:
        from src.data_pipeline.market_data import context_columns
        ctx_df = context_frame(cfg) if ctx_df is None else ctx_df
        ctx_names = list(ctx_names) if ctx_names else context_columns(cfg, ctx_df)
        aligned = ctx_df.reindex(columns=ctx_names).reindex(pd.DatetimeIndex(dates))
        ctx_raw = torch.tensor(aligned.to_numpy(dtype=np.float64), dtype=torch.float32)

    return {
        "tickers": tickers, "dates": dates, "x_raw": x_raw, "present": present,
        "node_feature_names": cols, "nodes": nodes_full, "ret1": ret1, "vol": vol,
        "edge_index": torch.tensor(ei, dtype=torch.long),
        "edge_weight": torch.tensor(ew, dtype=torch.float32),
        "directed_pairs": [(pos[u], pos[v]) for u, v in pair_names], "pair_names": pair_names,
        "pair_feat": _pair_features(cfg, ret1, pair_names),
        "pair_feature_names": list(cfg.gnn.pair_features),
        "ctx_raw": ctx_raw, "ctx_names": ctx_names, "ctx_pca": cfg.gnn.ctx_pca if ctx_raw is not None else 0,
    }


def compute_stats(raw: dict, train_dates: set) -> dict:
    """Train-period, present-only normalisation statistics."""
    dates = raw["dates"]
    mask_t = torch.tensor([d in train_dates for d in dates])
    x = raw["x_raw"][mask_t]                       # [T_tr, N, F]
    keep = raw["present"][mask_t]                  # [T_tr, N]
    flat = x[keep]                                 # [n, F]
    flat = flat[~torch.isnan(flat).any(-1)]
    stats = {"mean": flat.mean(dim=0), "std": flat.std(dim=0).clamp_min(1e-8)}
    if raw["ctx_raw"] is not None:
        c = raw["ctx_raw"][mask_t].numpy()
        stats["ctx_mean"] = torch.tensor(np.nan_to_num(np.nanmean(c, axis=0)), dtype=torch.float32)
        stats["ctx_std"] = torch.tensor(np.nan_to_num(np.nanstd(c, axis=0), nan=1.0), dtype=torch.float32).clamp_min(1e-8)
        k = raw.get("ctx_pca", 0)
        if k:
            z = _zscore_ctx(raw["ctx_raw"][mask_t], stats["ctx_mean"], stats["ctx_std"])
            _, _, vt = torch.linalg.svd(z - z.mean(0), full_matrices=False)
            comps = vt[:k].T.contiguous()                     # [M, k], fit on train days only
            proj = z @ comps
            stats["ctx_pca"] = comps
            stats["ctx_pca_mean"] = proj.mean(0)
            stats["ctx_pca_std"] = proj.std(0).clamp_min(1e-8)
    return stats


def _zscore_ctx(c: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num((c - mean) / std, nan=0.0).clamp(-CTX_CLIP, CTX_CLIP)


def apply_stats(cfg: Config, raw: dict, stats: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
    x = (raw["x_raw"] - stats["mean"]) / stats["std"]
    x = torch.nan_to_num(x, nan=0.0)
    x[~raw["present"]] = 0.0
    if cfg.gnn.history_start is not None:           # absent nodes are distinguishable
        x = torch.cat([x, raw["present"].float().unsqueeze(-1)], dim=-1)
    ctx = None
    if raw["ctx_raw"] is not None:
        ctx = _zscore_ctx(raw["ctx_raw"], stats["ctx_mean"], stats["ctx_std"])
        if "ctx_pca" in stats:
            ctx = ((ctx @ stats["ctx_pca"]) - stats["ctx_pca_mean"]) / stats["ctx_pca_std"]
            ctx = ctx.clamp(-CTX_CLIP, CTX_CLIP)
    return x, ctx


def _graph_data(cfg, raw, x, ctx, stats, train, val) -> GraphData:
    return GraphData(
        tickers=raw["tickers"], dates=raw["dates"], x=x,
        edge_index=raw["edge_index"], edge_weight=raw["edge_weight"],
        directed_pairs=raw["directed_pairs"], train=train, val=val,
        norm_mean=stats["mean"], norm_std=stats["std"],
        pair_feat=raw["pair_feat"], ctx=ctx,
        ctx_mean=stats.get("ctx_mean"), ctx_std=stats.get("ctx_std"),
        present=raw["present"], ret1=raw["ret1"], vol=raw["vol"],
        node_feature_names=raw["node_feature_names"] + (["present"] if cfg.gnn.history_start else []),
        ctx_names=raw["ctx_names"], pair_feature_names=raw["pair_feature_names"],
        extra_stats={k: v for k, v in stats.items() if k.startswith("ctx_pca")},
    )


def all_samples(cfg: Config, raw: dict) -> list[CascadeSample]:
    """Cascade samples whose source AND destination are listed at t and t+k."""
    nodes = raw["nodes"]
    shocks = shock_matrix(nodes, cfg).reindex(columns=raw["tickers"]).fillna(False)
    samples = build_samples(shocks, raw["pair_names"], cfg)
    present = pd.DataFrame(raw["present"].numpy(), index=pd.DatetimeIndex(raw["dates"]),
                           columns=raw["tickers"])
    if present.all().all():
        return samples
    k = cfg.gnn.cascade_window_days
    pos = {d: i for i, d in enumerate(present.index)}
    arr = present.to_numpy()
    col = {t: i for i, t in enumerate(raw["tickers"])}
    return [s for s in samples
            if arr[pos[s.date], col[s.src]] and arr[pos[s.date], col[s.dst]]
            and arr[min(pos[s.date] + k, len(arr) - 1), col[s.dst]]]


def cap_samples(cfg: Config, samples: list[CascadeSample], dates: list) -> list[CascadeSample]:
    """Drop samples whose k-day label window reaches past gnn.label_end_date,
    so neither training nor early stopping ever sees those prices (keeps the
    PPO test window out of GNN model selection)."""
    if not cfg.gnn.label_end_date:
        return samples
    end = pd.Timestamp(cfg.gnn.label_end_date)
    last = max(i for i, d in enumerate(dates) if d <= end)
    pos = {d: i for i, d in enumerate(dates)}
    k = cfg.gnn.cascade_window_days
    return [s for s in samples if pos[s.date] + k <= last]


def build_dataset(cfg: Config | None = None, raw: dict | None = None) -> GraphData:
    cfg = cfg or load_config()
    raw = raw or raw_inputs(cfg)
    samples = cap_samples(cfg, all_samples(cfg, raw), raw["dates"])
    train, val = temporal_split(samples, cfg)
    stats = compute_stats(raw, {s.date for s in train})
    x, ctx = apply_stats(cfg, raw, stats)
    return _graph_data(cfg, raw, x, ctx, stats, train, val)


def build_inference_inputs(cfg: Config, stats: dict, nodes: pd.DataFrame | None = None,
                           ctx_df: pd.DataFrame | None = None) -> GraphData:
    """Same tensors as build_dataset, normalised with PERSISTED stats (no labels)."""
    raw = raw_inputs(cfg, nodes=nodes, ctx_df=ctx_df, ctx_names=stats.get("ctx_names"))
    x, ctx = apply_stats(cfg, raw, stats)
    return _graph_data(cfg, raw, x, ctx, stats, [], [])
