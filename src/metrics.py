# agm/metrics.py

import pandas as pd
import numpy as np
from typing import Dict


def compute_performance_metrics(
    cum_pnl: pd.Series,
    drawdown: pd.Series,
    initial_capital: float,
    trading_days: int = 252
) -> Dict[str, float]:
    """
    Calculate performance metrics given cumulative PnL and drawdown.
    Returns a dict with keys:
      - gross_final_cum_pnl
      - annualized_return
      - max_drawdown
      - max_dd_duration_days
      - sharpe_ratio
    """
    equity = initial_capital + cum_pnl
    gross_final = cum_pnl.iloc[-1]
    n_days = len(equity)
    annualized_return = (equity.iloc[-1] / equity.iloc[0]) ** (trading_days / n_days) - 1
    max_drawdown = -drawdown.min()

    # Duration calculation
    is_underwater = drawdown != 0
    groups = (is_underwater != is_underwater.shift(1)).cumsum()
    durations = (
        pd.DataFrame({
            "under": is_underwater,
            "group": groups,
            "date": drawdown.index
        })
        .query("under")
        .groupby("group")["date"]
        .agg(lambda dr: (dr.max() - dr.min()).days)
    )
    max_dd_duration = int(durations.max()) if not durations.empty else 0

    daily_ret = equity.pct_change().dropna()
    sharpe = daily_ret.mean() / (daily_ret.std() + 1e-9) * np.sqrt(trading_days)

    return {
        "gross_final_cum_pnl": float(gross_final),
        "annualized_return": float(annualized_return),
        "max_drawdown": float(max_drawdown),
        "max_dd_duration_days": max_dd_duration,
        "sharpe_ratio": float(sharpe),
    }
