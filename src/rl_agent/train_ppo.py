"""Phase 6: PPO training (COMPUTE-HEAVY — run on the workstation).

    python scripts/train_ppo.py [--ablation full|no-gnn|no-sentiment|neither]
                                [--timesteps N]

One run per ablation arm produces src/rl_agent/logs/ppo_<ablation>/ with:
  model.zip, vecnormalize.pkl, train_meta.json, TensorBoard events.
The ablation arms (identical obs dims, blocks zeroed) are the capstone's
central experiment: does the hybrid state beat price-only?

Notes:
  * VecNormalize normalizes observations (running stats); reward left raw —
    differential Sharpe is already scale-controlled. Stats are SAVED and must
    be re-loaded at evaluation (evaluate.py handles it).
  * Reproducibility: set_global_seed + PPO(seed=...) + per-worker env seeds
    derived from the global seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.config import Config, load_config
from src.common.seeding import set_global_seed
from src.data_pipeline.state_builder import StateCache
from src.env.exchange_env import IndianExchangeEnv

LOGS_DIR = Path(__file__).resolve().parent / "logs"
ABLATIONS = {
    "full": {},
    "no-gnn": {"disable_gnn": True},
    "no-sentiment": {"disable_sentiment": True},
    "neither": {"disable_gnn": True, "disable_sentiment": True},
}


def make_vec_env(cfg: Config, state: StateCache, ablation: str, split: str = "train"):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    flags = ABLATIONS[ablation]

    def factory(rank: int):
        def _make():
            env = IndianExchangeEnv(cfg, split=split, state=state, **flags)
            env.reset(seed=cfg.project.seed + rank)
            return Monitor(env)
        return _make

    n = cfg.rl.n_envs if split == "train" else 1
    venv = DummyVecEnv([factory(i) for i in range(n)])
    return VecNormalize(venv, norm_obs=True, norm_reward=False, training=(split == "train"))


def train(cfg: Config | None = None, ablation: str = "full",
          timesteps: int | None = None) -> Path:
    from stable_baselines3 import PPO

    cfg = cfg or load_config()
    set_global_seed(cfg.project.seed)
    state = StateCache(cfg)
    if state.sentiment_source != "llm_cache" and ablation in ("full", "no-gnn"):
        print("WARNING: state.h5 sentiment is a neutral placeholder — run "
              "scripts/run_sentiment.py + scripts/build_state.py first, or this "
              "arm is equivalent to its no-sentiment counterpart.")

    run_dir = LOGS_DIR / f"ppo_{ablation}"
    run_dir.mkdir(parents=True, exist_ok=True)
    venv = make_vec_env(cfg, state, ablation)

    model = PPO(
        cfg.rl.policy,
        venv,
        seed=cfg.project.seed,
        n_steps=cfg.rl.n_steps,
        batch_size=cfg.rl.batch_size,
        gamma=cfg.rl.gamma,
        gae_lambda=cfg.rl.gae_lambda,
        learning_rate=cfg.rl.learning_rate,
        ent_coef=cfg.rl.ent_coef,
        tensorboard_log=str(run_dir),
        verbose=1,
    )
    total = timesteps or cfg.rl.total_timesteps
    model.learn(total_timesteps=total, progress_bar=True)

    model.save(run_dir / "model.zip")
    venv.save(str(run_dir / "vecnormalize.pkl"))
    (run_dir / "train_meta.json").write_text(json.dumps({
        "ablation": ablation,
        "timesteps": total,
        "seed": cfg.project.seed,
        "sentiment_source": state.sentiment_source,
        "state_config_sha256": str(state.config_sha256),
    }, indent=2))
    print(f"saved -> {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", choices=list(ABLATIONS), default="full")
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()
    train(ablation=args.ablation, timesteps=args.timesteps)


if __name__ == "__main__":
    main()
