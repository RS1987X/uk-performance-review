# agm/plotting_main.py

import argparse
import traceback
from pathlib import Path

import pandas as pd
from typing import Tuple
from src import compute_pnls
from src.plotting import (
    save_performance_table,
    plot_portfolio_vs_omx,
    plot_all_portfolios_vs_omx
)
from src import config


def load_omx_benchmark_from_prices(
    prices_csv_path: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Load the closing‐prices CSV (semicolon‐separated, 'date' column as index)
    and compute:
      - omx_cum_pnl: cumulative return series for `ticker`, starting at 0.0
      - omx_drawdown: drawdown series for `ticker`

    CSV must have:
      date;<ticker>;... 
      (date in YYYY-MM-DD format and closing prices in the column named `ticker`)

    Returns two pd.Series (indexed by date): (omx_cum_pnl, omx_drawdown).
    """
    # Read the CSV inside this function
    prices_df = pd.read_csv(
        prices_csv_path,
        sep=",",
        parse_dates=["Date"],
        index_col="Date"
    ).sort_index()

    omx_isin = "SE0002416156"
    omx_ticker = "^OMXSGI"
    if omx_ticker not in prices_df.columns:
        raise ValueError(f"OMX ticker '{omx_ticker}' not found in columns: {list(prices_df.columns)}")

    prices = prices_df[omx_ticker].dropna().sort_index()

    # Daily returns (fill NaN on first row with 0.0)
    daily_ret = prices.pct_change().fillna(0.0)

    # Cumulative return, starting at zero
    omx_cum_pnl = (1 + daily_ret).cumprod() - 1

    # Drawdown: difference between cum_pnl and its running maximum
    running_max = omx_cum_pnl.cummax()
    omx_drawdown = omx_cum_pnl - running_max

    return omx_cum_pnl, omx_drawdown


def main():
    """
    Orchestrate generating:
      1) Individual PnL vs. OMXSGI plots for each subportfolio
      2) An overall PnL vs. OMXSGI plot for the total portfolio
      3) A combined plot overlaying all subportfolios vs. OMXSGI
      4) Performance‐metrics tables for each subportfolio, overall portfolio, and OMXSGI

    Expects:
      - compute_pnls.main() returns (results_dict, overall_cum, overall_dd)
      - config.OMX_CSV points to a semicolon‐separated CSV with 'date','cum_pnl','drawdown'
      - config.INITIAL_CAPITAL is a float
      - config.START_DATE_STR is used internally by compute_pnls
      - config.OUTPUT_DIR is the root folder for all output PNGs
    """
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    print("▶ 1/3  Computing PnL curves for subportfolios and overall...")
    try:
        results_dict, overall_cum, overall_dd = compute_pnls.main()
    except Exception as e:
        print(f"⚠️  Error in compute_pnls: {e}")
        traceback.print_exc()
        return

    print("▶ 2/3  Loading OMXSGI benchmark series...")
    try:
        omx_cum, omx_dd = load_omx_benchmark_from_prices(config.PRICES_CSV)
        #omx_cum_sek, omx_dd_sek = config.INITIAL_SUB_PORTFOLIO_CAPITAL*omx_cum, config.INITIAL_SUB_PORTFOLIO_CAPITAL*omx_dd
    except Exception as e:
        print(f"⚠️  Error loading OMX benchmark: {e}")
        traceback.print_exc()
        return

    print("▶ 3/3  Generating plots and performance tables...")

    # 3a) Individual plots & tables for each subportfolio
    for name, (cum_pnl, drawdown) in results_dict.items():
        try:
            print(f"   • Subportfolio: {name}")
            # Plot subportfolio vs. OMX
            plot_path = plot_portfolio_vs_omx(
                name=name,
                cum_pnl=cum_pnl,
                drawdown=drawdown,
                omx_cum_pnl=omx_cum*config.INITIAL_SUB_PORTFOLIO_CAPITAL,
                omx_drawdown=omx_dd,
                output_dir=output_dir
            )
            print(f"     – Saved plot → {plot_path}")

            # Table of performance metrics for subportfolio
            table_path = save_performance_table(
                name=name,
                cum_pnl=cum_pnl,
                drawdown=drawdown,
                initial_capital=config.INITIAL_SUB_PORTFOLIO_CAPITAL,
                price_index_len=len(cum_pnl),
                output_dir=output_dir
            )
            print(f"     – Saved performance table → {table_path}")
        except Exception as e:
            print(f"⚠️  Error processing subportfolio '{name}': {e}")
            traceback.print_exc()

    # 3b) Overall portfolio vs. OMX
    try:
        overall_name = "Overall_Portfolio"
        print(f"   • {overall_name}")
        plot_path = plot_portfolio_vs_omx(
            name=overall_name,
            cum_pnl=overall_cum,
            drawdown=overall_dd,
            omx_cum_pnl=omx_cum*config.INITIAL_PORTFOLIO_CAPITAL,
            omx_drawdown=omx_dd*config.INITIAL_PORTFOLIO_CAPITAL,
            output_dir=output_dir
        )
        print(f"     – Saved plot → {plot_path}")

        table_path = save_performance_table(
            name=overall_name,
            cum_pnl=overall_cum,
            drawdown=overall_dd,
            initial_capital=config.INITIAL_PORTFOLIO_CAPITAL,
            price_index_len=len(omx_cum),
            output_dir=output_dir
        )
        print(f"     – Saved performance table → {table_path}")
    except Exception as e:
        print(f"⚠️  Error processing overall portfolio: {e}")
        traceback.print_exc()

    # 3c) Combined plot: all subportfolios vs. OMX
    try:
        combined_plot_path = plot_all_portfolios_vs_omx(
            results=results_dict,
            omx_cum_pnl=omx_cum*config.INITIAL_SUB_PORTFOLIO_CAPITAL,
            output_dir=output_dir
        )
        print(f"   • Combined PnL plot → {combined_plot_path}")
    except Exception as e:
        print(f"⚠️  Error creating combined PnL plot: {e}")
        traceback.print_exc()

    # 3d) Performance table for OMXSGI itself
    try:
        omx_table_path = save_performance_table(
            name="OMXSGI",
            cum_pnl=omx_cum,
            drawdown=omx_dd,
            initial_capital=config.INITIAL_SUB_PORTFOLIO_CAPITAL,            # use 1.0 as starting benchmark capital
            price_index_len=len(omx_cum),
            output_dir=output_dir
        )
        print(f"   • OMXSGI performance table → {omx_table_path}")
    except Exception as e:
        print(f"⚠️  Error creating OMXSGI performance table: {e}")
        traceback.print_exc()

    print("\n✅  All plots and tables generated.")


if __name__ == "__main__":
    main()
