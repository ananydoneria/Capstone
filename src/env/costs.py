"""Indian delivery-trade transaction cost model (NSE cash equities).

Rates live in config.yaml (env.costs) with citations in Sources.md.
Brokerage is assumed zero (discount-broker delivery). GST applies to the
exchange transaction charge and SEBI fee (statutory levies STT and stamp
duty are not GST-able). Slippage is applied to the FILL PRICE separately
(see exchange_env), not here.
"""

from __future__ import annotations

from src.common.config import CostsCfg


def buy_cost(notional: float, c: CostsCfg) -> float:
    """Total charges (INR) to buy `notional` INR of stock, excluding slippage."""
    gstable = notional * (c.exchange_txn_nse + c.sebi_turnover_fee)
    return (
        notional * c.stt_delivery
        + notional * c.stamp_duty_buy
        + gstable * (1 + c.gst_on_charges)
    )


def sell_cost(notional: float, c: CostsCfg) -> float:
    """Total charges (INR) to sell `notional` INR of stock, excluding slippage."""
    gstable = notional * (c.exchange_txn_nse + c.sebi_turnover_fee)
    return notional * c.stt_delivery + gstable * (1 + c.gst_on_charges)


def round_trip_bps(c: CostsCfg) -> float:
    """Approximate round-trip drag in basis points (diagnostics/reporting)."""
    return (buy_cost(1.0, c) + sell_cost(1.0, c) + 2 * c.slippage_bps / 1e4) * 1e4
