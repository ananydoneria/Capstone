"""Phase 5: the exchange sandbox — a Gymnasium env over pre-computed state.

Market physics:
  * Observation at day t: state-vector slice (features/GNN/sentiment, all
    computed from data <= t) + portfolio state. NEVER raw news or raw charts.
  * Action: Box(-1,1)^(N+1) logits -> softmax -> target weights over
    N assets + cash (long-only, fully self-financing).
  * Decisions made at close of t FILL AT OPEN OF t+1 (slippage + full Indian
    cost stack) — no same-bar fill bias. Equity marks at close of t+1.
  * Reward: differential Sharpe ratio (Moody & Saffell, per-step Sharpe
    gradient estimate) minus a penalty on drawdown beyond the configured cap.
  * Episodes: train split samples random `episode_length_days` windows
    (seeded); eval/test runs its full split once. Bankruptcy (<10% of initial
    equity) terminates.

The env only reads the in-RAM StateCache — no model inference, no I/O in the
loop (compute guardrail).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.common.config import Config, load_config
from src.data_pipeline.state_builder import StateCache
from src.env.portfolio import Portfolio

ACTION_LOGIT_SCALE = 5.0   # softmax(5*a), a in [-1,1]: allows ~full cash / full asset
BANKRUPTCY_FRACTION = 0.10


def split_range(cfg: Config, state: StateCache, split: str) -> tuple[int, int]:
    """Inclusive (lo, hi) day-index range of a walk-forward split."""
    if split == "train":
        mask = state.dates <= np.datetime64(cfg.rl.train_end)
    elif split == "test":
        mask = state.dates >= np.datetime64(cfg.rl.test_start)
    else:
        raise ValueError(f"unknown split {split!r}")
    idx = np.flatnonzero(mask)
    if len(idx) < 3:
        raise ValueError(f"split {split!r} too short")
    return int(idx[0]), int(idx[-1])


class IndianExchangeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Config | None = None,
        split: str = "train",              # "train" | "test"
        state: StateCache | None = None,   # share one cache across VecEnv workers
        disable_gnn: bool = False,         # ablation: zero the GNN block
        disable_sentiment: bool = False,   # ablation: zero the sentiment block
    ):
        super().__init__()
        self.cfg = cfg or load_config()
        self.state = state or StateCache(self.cfg)
        self.split = split
        self.disable_gnn = disable_gnn
        self.disable_sentiment = disable_sentiment

        self.lo, self.hi = split_range(self.cfg, self.state, split)

        self.n_assets = len(self.state.tickers)
        n_feat = self.state.features.shape[1] * self.state.features.shape[2]
        n_gnn = self.state.gnn.shape[1]
        obs_dim = n_feat + n_gnn + self.n_assets + self.n_assets + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_assets + 1,), np.float32)

    # ------------------------------------------------------------------ helpers
    def _obs(self) -> np.ndarray:
        t = self.t
        gnn = self.state.gnn[t]
        sent = self.state.sentiment[t]
        if self.disable_gnn:
            gnn = np.zeros_like(gnn)
        if self.disable_sentiment:
            sent = np.zeros_like(sent)
        close = self.state.close[t]
        eq = self.portfolio.equity(close)
        return np.concatenate([
            self.state.features[t].reshape(-1),
            gnn,
            sent,
            self.portfolio.weights(close),
            [self.portfolio.cash / eq if eq > 0 else 1.0],
        ]).astype(np.float32)

    def _differential_sharpe(self, r: float) -> float:
        eta = self.cfg.env.reward.sharpe_eta
        dA, dB = r - self._A, r * r - self._B
        denom = (self._B - self._A**2) ** 1.5
        d = (self._B * dA - 0.5 * self._A * dB) / denom if denom > 1e-12 else r
        self._A += eta * dA
        self._B += eta * dB
        return float(d)

    # ------------------------------------------------------------- gym interface
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        ep_len = self.cfg.rl.episode_length_days
        span = self.hi - self.lo
        if self.split == "train" and span > ep_len:
            start = self.lo + int(self.np_random.integers(0, span - ep_len + 1))
            self.end = min(start + ep_len, self.hi)
        else:
            start, self.end = self.lo, self.hi
        self.t = start
        self.portfolio = Portfolio(self.cfg.env.initial_cash_inr, self.n_assets)
        self._prev_equity = self.portfolio.equity(self.state.close[self.t])
        self._peak_equity = self._prev_equity
        self._A, self._B = 0.0, 0.0
        self.equity_curve = [self._prev_equity]
        self.turnover_total = 0.0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        logits = np.asarray(action, dtype=np.float64).clip(-1, 1) * ACTION_LOGIT_SCALE
        w = np.exp(logits - logits.max())
        w /= w.sum()
        target = w[: self.n_assets]          # w[-1] = cash bucket

        # fill next morning, mark at next close
        self.t += 1
        turnover = self.portfolio.rebalance(
            target, self.state.open[self.t], self.cfg.env.costs
        )
        self.turnover_total += turnover
        equity = self.portfolio.equity(self.state.close[self.t])
        r = equity / self._prev_equity - 1.0

        self._peak_equity = max(self._peak_equity, equity)
        drawdown = 1.0 - equity / self._peak_equity
        rc = self.cfg.env.reward
        reward = self._differential_sharpe(r) - rc.drawdown_penalty_coef * max(
            0.0, drawdown - rc.drawdown_cap
        )

        self._prev_equity = equity
        self.equity_curve.append(equity)
        terminated = equity < BANKRUPTCY_FRACTION * self.portfolio.initial_cash
        truncated = self.t >= self.end
        info = {"equity": equity, "return": r, "drawdown": drawdown, "turnover": turnover}
        return self._obs(), float(reward), terminated, truncated, info
