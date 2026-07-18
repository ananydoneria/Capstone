"""Phase 2.6: naive baselines vs the GNN on the SAME val samples.

Baselines score a val pair (date, src, dst) with no learning:
  1. train-period Pearson correlation of daily log returns (static structure)
  2. dst's previous-day 21d volatility (pure "volatile stocks shock more")
If the GNN can't beat these, it learned nothing beyond trivial priors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.seeding import set_global_seed
from src.data_pipeline.features import load_features
from src.models.gnn.graph_dataset import build_dataset
from src.models.gnn.model import PropagationGNN
from src.models.gnn.train import WEIGHTS_DIR, _batch_by_day, _epoch_scores, auc_score


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg.project.seed)
    data = build_dataset(cfg)
    features = load_features(cfg)
    labels = torch.tensor([s.label for s in data.val], dtype=torch.float32)

    # --- baseline 1: static train-period return correlation of the pair
    ret1 = features["logret_1"].unstack("ticker")
    train_dates = sorted({s.date for s in data.train})
    corr = ret1.loc[train_dates[0] : train_dates[-1]].corr()
    corr_scores = torch.tensor([corr.loc[s.src, s.dst] for s in data.val])

    # --- baseline 2: dst previous-day volatility
    vol = features[f"vol_{cfg.features.volatility_window}"].unstack("ticker").shift(1)
    vol_scores = torch.tensor([vol.loc[s.date, s.dst] for s in data.val])

    # --- trained GNN on identical samples
    model = PropagationGNN(in_dim=data.x.shape[-1], cfg=cfg)
    model.load_state_dict(torch.load(WEIGHTS_DIR / "propagation_gnn.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        gnn_logits, gnn_labels = _epoch_scores(model, data, _batch_by_day(data, data.val))

    print(f"val samples: {len(data.val)}  (positives {labels.mean():.1%})")
    print(f"baseline  train-corr AUC : {auc_score(labels, corr_scores):.4f}")
    print(f"baseline  dst-vol AUC    : {auc_score(labels, vol_scores):.4f}")
    print(f"GNN       val AUC        : {auc_score(gnn_labels, gnn_logits):.4f}")


if __name__ == "__main__":
    main()
