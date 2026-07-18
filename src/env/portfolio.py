"""Cash + holdings accounting for the exchange sandbox.

Fractional shares are permitted (standard backtest simplification; documented
limitation for the report). All costs are deducted from cash at fill time.
"""

from __future__ import annotations

import numpy as np

from src.common.config import CostsCfg
from src.env.costs import buy_cost, sell_cost


class Portfolio:
    def __init__(self, initial_cash: float, n_assets: int):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.shares = np.zeros(n_assets, dtype=np.float64)

    def equity(self, prices: np.ndarray) -> float:
        return float(self.cash + self.shares @ prices)

    def weights(self, prices: np.ndarray) -> np.ndarray:
        eq = self.equity(prices)
        return (self.shares * prices) / eq if eq > 0 else np.zeros_like(self.shares)

    def rebalance(
        self,
        target_weights: np.ndarray,   # [N] asset weights; residual to 1.0 = cash
        fill_prices: np.ndarray,      # [N] raw fill prices (pre-slippage)
        costs: CostsCfg,
    ) -> float:
        """Trade to target weights at fill prices. Returns turnover (INR traded).

        Sells execute before buys so proceeds fund purchases. Slippage worsens
        the effective price by slippage_bps in the trade direction. Buys are
        scaled down if cash (after costs) is insufficient — the portfolio can
        never go negative.
        """
        slip = costs.slippage_bps / 1e4
        equity = self.equity(fill_prices)
        target_value = target_weights * equity
        current_value = self.shares * fill_prices
        delta = target_value - current_value
        turnover = 0.0

        # --- sells
        for i in np.where(delta < 0)[0]:
            px = fill_prices[i] * (1 - slip)
            qty = min(-delta[i] / px, self.shares[i])
            notional = qty * px
            self.shares[i] -= qty
            self.cash += notional - sell_cost(notional, costs)
            turnover += notional

        # --- buys (scaled to available cash incl. per-notional cost)
        buy_idx = np.where(delta > 0)[0]
        if len(buy_idx):
            want = delta[buy_idx].sum()
            cost_rate = buy_cost(1.0, costs)
            affordable = max(self.cash, 0.0) / (1 + cost_rate)
            scale = min(1.0, affordable / want) if want > 0 else 0.0
            for i in buy_idx:
                notional = delta[i] * scale
                px = fill_prices[i] * (1 + slip)
                self.shares[i] += notional / px
                self.cash -= notional + buy_cost(notional, costs)
                turnover += notional
        return turnover
