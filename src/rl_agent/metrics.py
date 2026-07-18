"""Performance metrics over an equity curve (daily marks, INR)."""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def compute_metrics(equity_curve: list[float], turnover_total: float = 0.0) -> dict:
    eq = np.asarray(equity_curve, dtype=np.float64)
    rets = eq[1:] / eq[:-1] - 1.0
    std = rets.std(ddof=1) if len(rets) > 1 else 0.0
    downside = rets[rets < 0].std(ddof=1) if (rets < 0).sum() > 1 else 0.0
    peak = np.maximum.accumulate(eq)
    return {
        "total_return_pct": round((eq[-1] / eq[0] - 1) * 100, 3),
        "cagr_pct": round(((eq[-1] / eq[0]) ** (TRADING_DAYS / max(len(rets), 1)) - 1) * 100, 3),
        "sharpe": round(float(rets.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0, 3),
        "sortino": round(float(rets.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else 0.0, 3),
        "max_drawdown_pct": round(float((1 - eq / peak).max()) * 100, 3),
        "turnover_x_equity": round(turnover_total / eq[0], 3),
        "n_days": int(len(rets)),
    }
