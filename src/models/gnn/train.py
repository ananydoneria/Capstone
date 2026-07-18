"""GNN training: temporal split, BCE with class weighting, early stopping on
val AUC. Weights + normalization stats + config snapshot are saved to
``src/models/gnn/weights/`` so inference is exactly reproducible.

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


def _batch_by_day(data: GraphData, samples) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """Group samples per day: (day_idx, pairs [P,2], labels [P])."""
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
        out.append((day, pairs, labels))
    return out


def _epoch_scores(model, data, batches) -> tuple[torch.Tensor, torch.Tensor]:
    all_logits, all_labels = [], []
    for day, pairs, labels in batches:
        logits = model(data.x[day], data.edge_index, pairs)
        all_logits.append(logits)
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def train(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    set_global_seed(cfg.project.seed)
    data = build_dataset(cfg)

    train_b = _batch_by_day(data, data.train)
    val_b = _batch_by_day(data, data.val)
    n_tr = len(data.train)
    n_pos = sum(s.label for s in data.train)
    print(f"train samples: {n_tr} ({n_pos} positive, {n_pos / n_tr:.1%}) "
          f"over {len(train_b)} shock days; val samples: {len(data.val)} "
          f"over {len(val_b)} days")

    model = PropagationGNN(in_dim=data.x.shape[-1], cfg=cfg)
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.gnn.learning_rate, weight_decay=cfg.gnn.weight_decay
    )
    pos_weight = torch.tensor([(n_tr - n_pos) / max(n_pos, 1)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best = {"val_auc": -1.0, "epoch": -1, "state": None}
    patience_left = cfg.gnn.early_stopping_patience
    for epoch in range(cfg.gnn.epochs):
        model.train()
        total = 0.0
        for day, pairs, labels in train_b:
            opt.zero_grad()
            loss = loss_fn(model(data.x[day], data.edge_index, pairs), labels)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(labels)

        model.eval()
        with torch.no_grad():
            tr_logits, tr_labels = _epoch_scores(model, data, train_b)
            va_logits, va_labels = _epoch_scores(model, data, val_b)
        tr_auc, va_auc = auc_score(tr_labels, tr_logits), auc_score(va_labels, va_logits)

        if va_auc > best["val_auc"]:
            best = {"val_auc": va_auc, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
            patience_left = cfg.gnn.early_stopping_patience
        else:
            patience_left -= 1
        if epoch % 10 == 0 or patience_left == 0:
            print(f"epoch {epoch:3d}  loss {total / n_tr:.4f}  "
                  f"train AUC {tr_auc:.3f}  val AUC {va_auc:.3f}")
        if patience_left == 0:
            print(f"early stop at epoch {epoch} (best val AUC {best['val_auc']:.3f} "
                  f"@ epoch {best['epoch']})")
            break

    WEIGHTS_DIR.mkdir(exist_ok=True)
    torch.save(best["state"], WEIGHTS_DIR / "propagation_gnn.pt")
    torch.save({"mean": data.norm_mean, "std": data.norm_std}, WEIGHTS_DIR / "feature_norm.pt")
    meta = {
        "best_val_auc": round(best["val_auc"], 4),
        "best_epoch": best["epoch"],
        "seed": cfg.project.seed,
        "arch": cfg.gnn.arch,
        "in_dim": int(data.x.shape[-1]),
        "n_train": n_tr,
        "n_val": len(data.val),
        "train_pos_rate": round(n_pos / n_tr, 4),
        "shock_threshold": cfg.gnn.shock_return_threshold,
        "cascade_window_days": cfg.gnn.cascade_window_days,
    }
    (WEIGHTS_DIR / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved weights + norm stats + meta -> {WEIGHTS_DIR}")
    return meta
