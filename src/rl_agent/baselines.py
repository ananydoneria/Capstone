"""Phase 6.3: baseline strategies under the SAME exchange physics as PPO.

Each baseline replays a split using the identical primitives the env uses —
Portfolio.rebalance (full Indian cost stack + slippage), fill at open(t+1),
mark at close(t+1) — so comparisons are apples-to-apples:
  * buy_and_hold   — equal-weight buy on day 1, never trade again
  * equal_weight   — rebalance to equal weights every 21 trading days
  * momentum       — every 21 days, equal-weight the top 5 by 63-day momentum
    (mom_63 at day t is computed from data <= t: no lookahead)

Run:  python -m src.rl_agent.baselines
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.common.config import Config, load_config
from src.data_pipeline.state_builder import StateCache
from src.env.exchange_env import BANKRUPTCY_FRACTION, split_range
from src.env.portfolio import Portfolio
from src.rl_agent.metrics import compute_metrics

REBALANCE_EVERY = 21
MOMENTUM_TOP_K = 5
LOGS_DIR = Path(__file__).resolve().parent / "logs"


def _replay(cfg: Config, state: StateCache, split: str, weight_fn) -> dict:
    """weight_fn(state, t, step_i) -> target weights [N] or None (no trade)."""
    lo, hi = split_range(cfg, state, split)
    n = len(state.tickers)
    portfolio = Portfolio(cfg.env.initial_cash_inr, n)
    equity_curve = [portfolio.equity(state.close[lo])]
    turnover_total = 0.0
    for step_i, t in enumerate(range(lo, hi)):          # decide at t, fill at t+1
        target = weight_fn(state, t, step_i)
        if target is not None:
            turnover_total += portfolio.rebalance(target, state.open[t + 1], cfg.env.costs)
        equity = portfolio.equity(state.close[t + 1])
        equity_curve.append(equity)
        if equity < BANKRUPTCY_FRACTION * portfolio.initial_cash:
            break
    return compute_metrics(equity_curve, turnover_total)


def _equal(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def buy_and_hold(state: StateCache, t: int, i: int):
    return _equal(len(state.tickers)) if i == 0 else None


def equal_weight(state: StateCache, t: int, i: int):
    return _equal(len(state.tickers)) if i % REBALANCE_EVERY == 0 else None


def momentum(state: StateCache, t: int, i: int):
    if i % REBALANCE_EVERY != 0:
        return None
    scores = state.features[t, :, state.feature_index["mom_63"]]
    top = np.argsort(scores)[-MOMENTUM_TOP_K:]
    w = np.zeros(len(state.tickers))
    w[top] = 1.0 / MOMENTUM_TOP_K
    return w


def run_all(cfg: Config | None = None, split: str = "test") -> dict[str, dict]:
    cfg = cfg or load_config()
    state = StateCache(cfg)
    results = {
        name: _replay(cfg, state, split, fn)
        for name, fn in [("buy_and_hold", buy_and_hold),
                         ("equal_weight", equal_weight),
                         ("momentum", momentum)]
    }
    LOGS_DIR.mkdir(exist_ok=True)
    (LOGS_DIR / f"baselines_{split}.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    for name, m in run_all().items():
        print(f"{name:13s} total {m['total_return_pct']:+7.2f}%  sharpe {m['sharpe']:+.2f}  "
              f"maxDD {m['max_drawdown_pct']:.1f}%  turnover {m['turnover_x_equity']:.1f}x")
