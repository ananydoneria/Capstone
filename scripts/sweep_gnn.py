"""Small GNN hyperparameter sweep (capacity x dropout x weight decay).

Each run is seconds on CPU. Prints a ranked table; also re-runs the best
config across 3 seeds to expose val-AUC variance (selection honesty on a
565-sample val set). config.yaml is NOT modified — update it manually with
the winner, then regenerate final weights via scripts/train_gnn.py.
"""

from __future__ import annotations

import contextlib
import copy
import io
import itertools
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import Config
from src.models.gnn.train import train


def variant(raw: dict, **gnn_updates) -> Config:
    r = copy.deepcopy(raw)
    r["gnn"].update(gnn_updates)
    return Config.model_validate(r)


def quiet_train(cfg: Config) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        return train(cfg)


def main() -> None:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    grid = list(itertools.product([16, 32, 64], [0.2, 0.4], [0.0, 0.001]))
    results = []
    for h, dr, wd in grid:
        meta = quiet_train(variant(raw, hidden_dim=h, dropout=dr, weight_decay=wd))
        results.append((meta["best_val_auc"], h, dr, wd, meta["best_epoch"]))
        print(f"h={h:2d} dropout={dr} wd={wd:<6} -> val AUC {meta['best_val_auc']:.4f} "
              f"@ epoch {meta['best_epoch']}")

    results.sort(reverse=True)
    print("\ntop 3:")
    for auc, h, dr, wd, ep in results[:3]:
        print(f"  val AUC {auc:.4f}  h={h} dropout={dr} wd={wd} (epoch {ep})")

    best_auc, h, dr, wd, _ = results[0]
    print(f"\nseed robustness of winner (h={h}, dropout={dr}, wd={wd}):")
    for seed in (42, 7, 2026):
        cfg = variant(raw, hidden_dim=h, dropout=dr, weight_decay=wd)
        cfg = Config.model_validate(
            {**cfg.model_dump(), "project": {**cfg.project.model_dump(), "seed": seed}}
        )
        meta = quiet_train(cfg)
        print(f"  seed {seed}: val AUC {meta['best_val_auc']:.4f} @ epoch {meta['best_epoch']}")


if __name__ == "__main__":
    main()
