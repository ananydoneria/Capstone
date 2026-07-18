"""Phase 5 gate: cost math, SB3 env check, determinism, physics invariants."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.data_pipeline.state_builder import state_path
from src.env.costs import buy_cost, round_trip_bps, sell_cost
from src.env.portfolio import Portfolio

cfg = load_config()
needs_state = pytest.mark.skipif(
    not state_path(cfg).exists(), reason="run scripts/build_state.py first"
)


def test_cost_model_rates():
    c = cfg.env.costs
    # buy 1L INR: STT 100 + stamp 15 + txn 2.97 + sebi 0.10 + 18% GST on the last two
    assert buy_cost(100_000, c) == pytest.approx(100 + 15 + (2.97 + 0.10) * 1.18, rel=1e-6)
    assert sell_cost(100_000, c) == pytest.approx(100 + (2.97 + 0.10) * 1.18, rel=1e-6)
    assert 20 < round_trip_bps(c) < 50  # sanity: Indian delivery round trip ~0.3%


def test_portfolio_never_goes_negative():
    p = Portfolio(initial_cash=100_000, n_assets=2)
    prices = np.array([100.0, 50.0])
    p.rebalance(np.array([0.7, 0.3]), prices, cfg.env.costs)
    assert p.cash >= 0
    eq = p.equity(prices)
    assert eq < 100_000  # costs were paid
    assert eq > 99_000
    # sell everything back
    p.rebalance(np.array([0.0, 0.0]), prices, cfg.env.costs)
    assert p.cash >= 0 and np.allclose(p.shares, 0, atol=1e-9)


def test_rebalance_hits_target_weights():
    p = Portfolio(initial_cash=1_000_000, n_assets=3)
    prices = np.array([2500.0, 700.0, 120.0])
    target = np.array([0.5, 0.3, 0.1])
    p.rebalance(target, prices, cfg.env.costs)
    assert np.allclose(p.weights(prices), target, atol=0.01)


@needs_state
def test_sb3_env_check():
    from stable_baselines3.common.env_checker import check_env

    from src.env.exchange_env import IndianExchangeEnv

    check_env(IndianExchangeEnv(cfg, split="train"), warn=True)


@needs_state
def test_deterministic_replay():
    """Same seed + same actions -> bit-identical equity trajectory."""
    from src.env.exchange_env import IndianExchangeEnv

    def rollout():
        env = IndianExchangeEnv(cfg, split="train")
        env.reset(seed=cfg.project.seed)
        env.action_space.seed(cfg.project.seed)
        for _ in range(40):
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                break
        return env.equity_curve

    assert rollout() == rollout()


@needs_state
def test_near_cash_action_stays_near_initial_equity():
    """A cash-dominated softmax action trades only dust (softmax can't emit
    exact zeros), so turnover and equity impact must be negligible."""
    from src.env.exchange_env import IndianExchangeEnv

    env = IndianExchangeEnv(cfg, split="train")
    env.reset(seed=cfg.project.seed)
    all_cash = np.full(env.action_space.shape, -1.0, dtype=np.float32)
    all_cash[-1] = 1.0  # cash bucket dominates the softmax
    _, _, _, _, info = env.step(all_cash)
    assert info["turnover"] < 2e-3 * cfg.env.initial_cash_inr
    assert info["equity"] == pytest.approx(cfg.env.initial_cash_inr, rel=1e-4)


@needs_state
def test_ablation_blocks_zeroed():
    from src.env.exchange_env import IndianExchangeEnv

    full_env = IndianExchangeEnv(cfg, split="train")
    abl_env = IndianExchangeEnv(cfg, split="train", disable_gnn=True)
    obs_f, _ = full_env.reset(seed=1)
    obs_a, _ = abl_env.reset(seed=1)
    n_feat = 15 * 6
    n_gnn = full_env.state.gnn.shape[1]
    assert obs_f.shape == obs_a.shape  # identical dims across ablation arms
    assert np.abs(obs_f[n_feat : n_feat + n_gnn]).sum() > 0
    assert np.abs(obs_a[n_feat : n_feat + n_gnn]).sum() == 0
