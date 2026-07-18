"""Phase 6.4: out-of-sample evaluation of trained PPO arms on the test split.

Loads model.zip + vecnormalize.pkl per ablation arm, replays the test window
deterministically, and writes metrics + the equity curve next to the model.

    python scripts/evaluate.py [--ablation full|no-gnn|no-sentiment|neither|all]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.common.config import Config, load_config
from src.common.seeding import set_global_seed
from src.data_pipeline.state_builder import StateCache
from src.env.exchange_env import IndianExchangeEnv
from src.rl_agent.metrics import compute_metrics
from src.rl_agent.train_ppo import ABLATIONS, LOGS_DIR


def evaluate_arm(cfg: Config, state: StateCache, ablation: str) -> dict | None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    run_dir = LOGS_DIR / f"ppo_{ablation}"
    if not (run_dir / "model.zip").exists():
        print(f"{ablation}: no trained model at {run_dir} — skipped")
        return None

    def _make():
        env = IndianExchangeEnv(cfg, split="test", state=state, **ABLATIONS[ablation])
        env.reset(seed=cfg.project.seed)
        return Monitor(env)

    venv = VecNormalize.load(str(run_dir / "vecnormalize.pkl"), DummyVecEnv([_make]))
    venv.training = False          # freeze normalization stats
    venv.norm_reward = False
    model = PPO.load(run_dir / "model.zip", env=venv)

    obs = venv.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        done = bool(dones[0])
    inner: IndianExchangeEnv = venv.venv.envs[0].unwrapped
    metrics = compute_metrics(inner.equity_curve, inner.turnover_total)

    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savetxt(run_dir / "test_equity_curve.csv", np.asarray(inner.equity_curve),
               header="equity_inr", comments="")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", default="all",
                        choices=[*ABLATIONS, "all"])
    args = parser.parse_args()
    cfg = load_config()
    set_global_seed(cfg.project.seed)
    state = StateCache(cfg)
    arms = list(ABLATIONS) if args.ablation == "all" else [args.ablation]
    for arm in arms:
        m = evaluate_arm(cfg, state, arm)
        if m:
            print(f"{arm:13s} total {m['total_return_pct']:+7.2f}%  "
                  f"sharpe {m['sharpe']:+.2f}  maxDD {m['max_drawdown_pct']:.1f}%")


if __name__ == "__main__":
    main()
