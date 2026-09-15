"""GNN training: temporal split, BCE with class weighting, early stopping on
val AUC. Weights + normalization stats + config snapshot are saved to
``src/models/gnn/weights/`` so inference is exactly reproducible.

``fit`` is the reusable core (walk-forward validation calls it per fold);
``train`` is the entrypoint that fits on the configured split and saves.

Run:  python scripts/train_gnn.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from src.common.config import Config, load_config
from src.common.seeding import set_global_seed
from src.models.gnn.graph_dataset import GraphData, build_dataset
from src.models.gnn.model import PropagationGNN

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def auc_score(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """Rank-based AUC (Mann-Whitney), dependency-free."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().sum()
    ties = (pos.unsqueeze(1) == neg.unsqueeze(0)).float().sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def make_model(cfg: Config, data: GraphData) -> PropagationGNN:
    return PropagationGNN(in_dim=data.x.shape[-1], cfg=cfg, pair_dim=data.pair_dim, ctx_dim=data.ctx_dim)


def _batch_by_day(data: GraphData, samples) -> list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Group samples per day: (day_idx, pairs [P,2], labels [P], pair_ids [P])."""
    pos = {t: i for i, t in enumerate(data.tickers)}
    by_day: dict[int, list] = {}
    for s in samples:
        by_day.setdefault(data.day_index(s.date), []).append(
            (pos[s.src], pos[s.dst], s.label)
        )
    out = []
    for day, rows in sorted(by_day.items()):
        pairs = torch.tensor([[r[0], r[1]] for r in rows], dtype=torch.long)
        labels = torch.tensor([r[2] for r in rows], dtype=torch.float32)
        pair_ids = torch.tensor([data.pair_id(r[0], r[1]) for r in rows], dtype=torch.long) \
            if data.directed_pairs else torch.zeros(len(rows), dtype=torch.long)
        out.append((day, pairs, labels, pair_ids))
    return out


def forward_day(model: PropagationGNN, data: GraphData, day: int, pairs: torch.Tensor,
                pair_ids: torch.Tensor | None, window: int, x: torch.Tensor | None = None) -> torch.Tensor:
    """One day's logits for the given pairs, wiring in whichever optional
    inputs (edge weights, pair features, market context) the model uses."""
    xs = data.x if x is None else x
    from src.models.gnn.graph_dataset import window_features
    pf = data.pair_feat[day, pair_ids] if (model.pair_dim and pair_ids is not None) else None
    ctx = data.ctx[day] if model.ctx_dim else None
    ew = data.edge_weight if model.uses_edge_weight else None
    return model(window_features(xs, day, window), data.edge_index, pairs,
                 edge_weight=ew, pair_feat=pf, ctx=ctx)


def forward_all_pairs(model: PropagationGNN, data: GraphData, day: int, window: int,
                      x: torch.Tensor | None = None) -> torch.Tensor:
    """Logits for every supervision pair, in supervision_pairs() order."""
    pairs = torch.tensor(data.directed_pairs, dtype=torch.long)
    return forward_day(model, data, day, pairs, torch.arange(len(pairs)), window, x=x)


def _epoch_scores(model, data, batches, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    all_logits, all_labels = [], []
    for b in batches:
        day, pairs, labels = b[0], b[1], b[2]
        pair_ids = b[3] if len(b) > 3 else None
        all_logits.append(forward_day(model, data, day, pairs, pair_ids, window))
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def fit(cfg: Config, data: GraphData, train_samples, val_samples, verbose: bool = True) -> dict:
    """Train on train_samples, early-stop on val_samples AUC. Returns the best
    state and bookkeeping; does not touch disk."""
    set_global_seed(cfg.project.seed)
    train_b = _batch_by_day(data, train_samples)
    val_b = _batch_by_day(data, val_samples)
    n_tr = len(train_samples)
    n_pos = sum(s.label for s in train_samples)
    if verbose:
        print(f"train samples: {n_tr} ({n_pos} positive, {n_pos / n_tr:.1%}) "
              f"over {len(train_b)} shock days; val samples: {len(val_samples)} "
              f"over {len(val_b)} days")

    model = make_model(cfg, data)
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.gnn.learning_rate, weight_decay=cfg.gnn.weight_decay
    )
    pos_weight = torch.tensor([(n_tr - n_pos) / max(n_pos, 1)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    window = cfg.gnn.temporal_window_days
    best = {"val_auc": -1.0, "epoch": -1, "state": None, "train_auc": float("nan")}
    patience_left = cfg.gnn.early_stopping_patience
    for epoch in range(cfg.gnn.epochs):
        model.train()
        total = 0.0
        for day, pairs, labels, pair_ids in train_b:
            opt.zero_grad()
            loss = loss_fn(forward_day(model, data, day, pairs, pair_ids, window), labels)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(labels)

        model.eval()
        with torch.no_grad():
            tr_logits, tr_labels = _epoch_scores(model, data, train_b, window)
            va_logits, va_labels = _epoch_scores(model, data, val_b, window)
        tr_auc, va_auc = auc_score(tr_labels, tr_logits), auc_score(va_labels, va_logits)

        if va_auc > best["val_auc"]:
            best = {"val_auc": va_auc, "epoch": epoch, "train_auc": tr_auc,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
            patience_left = cfg.gnn.early_stopping_patience
        else:
            patience_left -= 1
        if verbose and (epoch % 10 == 0 or patience_left == 0):
            print(f"epoch {epoch:3d}  loss {total / n_tr:.4f}  "
                  f"train AUC {tr_auc:.3f}  val AUC {va_auc:.3f}")
        if patience_left == 0:
            if verbose:
                print(f"early stop at epoch {epoch} (best val AUC {best['val_auc']:.3f} "
                      f"@ epoch {best['epoch']})")
            break
    best.update(n_train=n_tr, n_pos=n_pos, n_val=len(val_samples))
    return best


def save_artifacts(cfg: Config, data: GraphData, best: dict, weights_dir: Path | None = None) -> dict:
    wdir = weights_dir or WEIGHTS_DIR
    wdir.mkdir(exist_ok=True)
    torch.save(best["state"], wdir / "propagation_gnn.pt")
    norm = {"mean": data.norm_mean, "std": data.norm_std, **(data.extra_stats or {})}
    if data.ctx_mean is not None:
        norm.update(ctx_mean=data.ctx_mean, ctx_std=data.ctx_std, ctx_names=list(data.ctx_names))
    torch.save(norm, wdir / "feature_norm.pt")
    n_tr, n_pos = best["n_train"], best["n_pos"]
    meta = {
        "best_val_auc": round(best["val_auc"], 4),
        "best_epoch": best["epoch"],
        "seed": cfg.project.seed,
        "arch": cfg.gnn.arch,
        "in_dim": int(data.x.shape[-1]),
        "pair_dim": data.pair_dim,
        "ctx_dim": data.ctx_dim,
        "n_train": n_tr,
        "n_val": best["n_val"],
        "train_pos_rate": round(n_pos / n_tr, 4),
        "train_start_date": cfg.gnn.train_start_date,
        "history_start": cfg.gnn.history_start,
        "market_context": cfg.gnn.market_context,
        "breadth": cfg.gnn.breadth,
        "pair_features": list(cfg.gnn.pair_features),
        "ctx_pca": cfg.gnn.ctx_pca,
        "label_end_date": cfg.gnn.label_end_date,
        "val_start": str(min(s.date for s in data.val).date()) if data.val else None,
        "val_end": str(max(s.date for s in data.val).date()) if data.val else None,
        "shock_z": cfg.gnn.shock_z,
        "shock_min_move": cfg.gnn.shock_min_move,
        "bidirectional_supervision": cfg.gnn.bidirectional_supervision,
        "cascade_window_days": cfg.gnn.cascade_window_days,
    }
    (wdir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_stats(weights_dir: Path | None = None) -> dict:
    return torch.load((weights_dir or WEIGHTS_DIR) / "feature_norm.pt", weights_only=True)


def train(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    set_global_seed(cfg.project.seed)
    data = build_dataset(cfg)
    best = fit(cfg, data, data.train, data.val)
    meta = save_artifacts(cfg, data, best)
    print(f"saved weights + norm stats + meta -> {WEIGHTS_DIR}")
    return meta
