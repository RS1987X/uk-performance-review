# agm/compute_pnls.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from src import config
from src.data_loader import load_prices, load_transactions, load_start_positions, infer_currency_map
from src.metrics import compute_performance_metrics


def build_daily_positions(
    tx: pd.DataFrame,
    start_pos_df: pd.DataFrame,
    prices: pd.DataFrame,
    currency_map: Dict[str, str],
    start_date_str: str,
) -> Dict[str, pd.DataFrame]:
    """
    For each portfolio:
      1. Pivot buys/sells by date & ticker → daily signed flows
      2. Incorporate starting positions (on start_date)
      3. Cumulatively sum to get daily positions
      4. Convert positions into SEK using the currency_map & price data

    Returns a dict: { portfolio_name → positions_df (dates × tickers) } in SEK units.
    """
    positions_per_port = {}
    price_dates = prices.index  # dates only

    # 1) Compute signed shares for each trade
    tx_copy = tx.copy()
    tx_copy["SignedShares"] = 0
    is_buy = tx_copy["Transaktionstyp"] == "KÖPT"
    is_sell = tx_copy["Transaktionstyp"] == "SÅLT"
    tx_copy.loc[is_buy, "SignedShares"] = tx_copy.loc[is_buy, "Antal"]
    tx_copy.loc[is_sell, "SignedShares"] = -tx_copy.loc[is_sell, "Antal"]

    portfolios = {}
    # all_port_names = set(tx_copy["Portfolio"].dropna().unique()).union(
    #     set(start_pos_df["Portfolio"].dropna().unique())
    # )

    all_port_names = set(config.SUB_PORTFOLIO_NAMES)

    for port_name in sorted(all_port_names):
        grp = tx_copy[tx_copy["subportfolio"] == port_name]
        # Pivot buy/sell separately
        buys = (
            grp[grp["Transaktionstyp"] == "KÖPT"]
            .pivot_table(index="Date", columns="Identifying name", values="Antal", aggfunc="sum")
        )
        sells = (
            grp[grp["Transaktionstyp"] == "SÅLT"]
            .pivot_table(index="Date", columns="Identifying name", values="Antal", aggfunc="sum")
        )
        signed_buys = buys.fillna(0).reindex(price_dates, fill_value=0)
        signed_sells = -sells.fillna(0).reindex(price_dates, fill_value=0)

        # Include starting positions on start_date
        pos_start = start_pos_df[start_pos_df["subportfolio"] == port_name]
        all_tickers = set(signed_buys.columns).union(signed_sells.columns).union(set(pos_start["Name"]))
        signed_buys = signed_buys.reindex(columns=all_tickers, fill_value=0)
        signed_sells = signed_sells.reindex(columns=all_tickers, fill_value=0)
        pos_full = signed_buys.add(signed_sells, fill_value=0)

        if not pos_start.empty:
            sd = pd.to_datetime(start_date_str).date()
            if sd not in pos_full.index:
                raise KeyError(f"Start date {sd} not in price index for sub portfolio '{port_name}'.")
            start_row = pd.Series(
                {row["Name"]: row["Shares"] for _, row in pos_start.iterrows()},
                index=all_tickers,
            ).fillna(0)
            pos_full.loc[sd, :] += start_row

        # Cumulative sum → “positions as of each date”
        cum_pos = pos_full.cumsum()
        positions_per_port[port_name] = cum_pos

    return positions_per_port


def mark_to_market(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    currency_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Given a positions DataFrame (dates × tickers) and full price DataFrame (dates × columns),
    convert all non‐SEK tickers into SEK (multiplying by FX), then multiply positions × prices
    → a DataFrame of mark‐to‐market value. Missing FX columns will raise a KeyError.
    """
    # Build a price_in_sek DataFrame with the same shape as positions
    price_sek = pd.DataFrame(index=prices.index, columns=positions.columns, dtype=float)
    for ticker in positions.columns:
        ccy = currency_map.get(ticker, "SEK")
        if ccy == "SEK":
            price_sek[ticker] = prices.get(ticker, pd.NA)
        else:
            fx_col = f"{ccy}SEK=X"
            if fx_col not in prices.columns:
                raise KeyError(f"Missing FX rate for {ccy}: expected '{fx_col}'.")
            price_sek[ticker] = prices[ticker] * prices[fx_col]
    # Multiply positions (shares) × price → market value
    return positions * price_sek.fillna(0)


def compute_unrealized_pnl(
    positions: pd.DataFrame,
    price_sek: pd.DataFrame
) -> pd.DataFrame:
    """
    Unrealized PnL_ticker = yesterday's_position × (today_price - yesterday_price).
    positions and price_sek must have a datetime index aligned and same columns.
    """
    # shift positions by one day so that PnL on day t = position(t-1) * price.diff(t)
    realized = None  # (we’ll compute realized separately)
    per_ticker_unrealized = positions.shift(1) * price_sek.diff()
    return per_ticker_unrealized.fillna(0)


def compute_realized_pnl(
    tx: pd.DataFrame
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    For each trade date & ticker, compute same‐day realized PnL:
    pnl_same_day = matched_qty * (avg_sell_price - avg_buy_price).
    Returns:
      - realized_pnl_by_portfolio_date (pd.Series indexed by date, sum over all tickers/portfolios)
      - per_ticker_realized_df (DataFrame: dates × tickers, realized‐by‐ticker on that date)
    """
    tx_copy = tx.copy()
    tx_copy["signed_qty"] = tx_copy["Antal"].where(
        tx_copy["Transaktionstyp"] == "SÅLT", -tx_copy["Antal"]
    )
    # Group by normalized date & ticker
    grouped = (
        tx_copy[tx_copy["Transaktionstyp"].isin(["KÖPT", "SÅLT"])]
        .groupby([pd.to_datetime(tx_copy["Date"]).dt.normalize(), tx_copy["Identifying name"]])
    )

    # We’ll collect results into a per‐ticker DataFrame
    dates = pd.to_datetime(tx_copy["Date"]).dt.normalize().sort_values().unique()
    tickers = tx_copy["Identifying name"].unique()
    per_ticker_realized = pd.DataFrame(0.0, index=dates, columns=tickers)
    realized_pnl_series = pd.Series(0.0, index=dates)

    for (dt, ticker), sub in grouped:
        buys = sub[sub["Transaktionstyp"] == "KÖPT"]
        sells = sub[sub["Transaktionstyp"] == "SÅLT"]
        if not buys.empty and not sells.empty:
            total_bought = buys["Antal"].sum()
            total_sold = sells["Antal"].sum()
            matched_qty = min(total_bought, total_sold)
            avg_buy_price = (buys["Antal"] * buys["Pris SEK"].abs()).sum() / total_bought
            avg_sell_price = (sells["Antal"] * sells["Pris SEK"]).sum() / total_sold
            pnl_same_day = matched_qty * (avg_sell_price - avg_buy_price)
            per_ticker_realized.at[dt, ticker] += pnl_same_day
            realized_pnl_series.at[dt] += pnl_same_day

    return realized_pnl_series, per_ticker_realized


def aggregate_portfolio_pnl(
    tx: pd.DataFrame,
    prices: pd.DataFrame,
    start_pos_df: pd.DataFrame,
    initial_capital: float,
    start_date_str: str
) -> Tuple[Dict[str, Tuple[pd.Series, pd.Series]], pd.Series, pd.Series]:
    """
    Orchestrates:
      1. Infer currency_map
      2. Build daily positions per portfolio
      3. Compute mark-to-market prices (price_sek) for each portfolio's positions
      4. Compute per‐ticker unrealized & per‐ticker realized PnL
      5. Sum across tickers → daily PnL per portfolio
      6. Compute cum_pnl & drawdown per portfolio
      7. Also compute UK aggregate PnL & drawdown
    Returns:
      - results_per_portfolio: { portfolio_name → (cum_pnl Series, drawdown Series) }
      - uk_cum_pnl: pd.Series of aggregate UK daily cum PnL
      - uk_drawdown: pd.Series of aggregate UK daily drawdown
    """
    # 1) Infer currency map
    currency_map = infer_currency_map(tx, start_pos_df)

    # 2) Build daily positions
    positions_per_port = build_daily_positions(tx, start_pos_df, prices, currency_map, start_date_str)

    results = {}
    total_unrealized_by_ticker = None
    total_pnl_by_ticker = None

    # We’ll accumulate UK daily PnL so we can compute aggregate curves
    uk_daily_unrealized = None
    uk_daily_realized = None

    for port_name, positions in positions_per_port.items():
        # 3) Build price_in_sek for this portfolio
        price_sek = pd.DataFrame(index=prices.index, columns=positions.columns, dtype=float)
        for ticker in positions.columns:
            ccy = currency_map.get(ticker, "SEK")
            if ccy == "SEK":
                price_sek[ticker] = prices.get(ticker, pd.NA)
            else:
                fx_col = f"{ccy}SEK=X"
                if fx_col not in prices.columns:
                    raise KeyError(f"Missing FX rate for {ccy}: expected '{fx_col}'.")
                price_sek[ticker] = prices[ticker] * prices[fx_col]
        price_sek = price_sek.ffill().fillna(0)

        # 4) Unrealized PnL per ticker
        per_ticker_unrealized = compute_unrealized_pnl(positions, price_sek)

        # 5) Realized PnL per date & ticker
        trades_port = tx[tx["subportfolio"] == port_name]
        daily_realized_series, per_ticker_realized = compute_realized_pnl(trades_port)

        # 6) Per‐ticker total daily PnL = unrealized + realized_by_ticker
        per_ticker_pnl = per_ticker_unrealized.add(per_ticker_realized, fill_value=0)

        # Accumulate per‐ticker across portfolios if you want an overall table
        if total_pnl_by_ticker is None:
            total_pnl_by_ticker = per_ticker_pnl.copy()
        else:
            total_pnl_by_ticker = total_pnl_by_ticker.add(per_ticker_pnl, fill_value=0)

        # 7) Build portfolio‐level daily PnL
        daily_unrealized = per_ticker_unrealized.sum(axis=1).fillna(0)
        daily_total_pnl = daily_unrealized.add(daily_realized_series, fill_value=0)

        # 8) Add dividends & withholding tax for this portfolio
        divs = (
            trades_port[trades_port["Transaktionstyp"] == "UTDELNING"]
            .groupby(pd.to_datetime(trades_port["Date"]).dt.normalize())["Belopp SEK (ex courtage)"]
            .sum()
            .reindex(prices.index, fill_value=0)
        )
        tax = -(
            trades_port[trades_port["Transaktionstyp"] == "UTL KÄLLSKATT"]
            .groupby(pd.to_datetime(trades_port["Date"]).dt.normalize())["Belopp SEK (ex courtage)"]
            .sum()
            .abs()
            .reindex(prices.index, fill_value=0)
        )
        daily_total_pnl = daily_total_pnl.add(divs, fill_value=0).add(tax, fill_value=0)

        # 9) Cumulative PnL & drawdown
        cum_pnl = daily_total_pnl.cumsum()
        drawdown = cum_pnl - cum_pnl.cummax()
        results[port_name] = (cum_pnl, drawdown)

        # Accumulate UK daily PnL
        if uk_daily_unrealized is None:
            uk_daily_unrealized = daily_unrealized.copy()
            uk_daily_realized = daily_realized_series.copy()
        else:
            uk_daily_unrealized = uk_daily_unrealized.add(daily_unrealized, fill_value=0)
            uk_daily_realized = uk_daily_realized.add(daily_realized_series, fill_value=0)

    # 10) UK aggregate PnL
    uk_daily_total_pnl = uk_daily_unrealized.add(uk_daily_realized, fill_value=0)
    uk_cum_pnl = uk_daily_total_pnl.cumsum()
    uk_drawdown = uk_cum_pnl - uk_cum_pnl.cummax()

    return results, uk_cum_pnl, uk_drawdown


def main():
    # 1) Load data
    prices_df = load_prices(config.PRICES_CSV)
    tx_df = load_transactions(config.AUGMENTET_AND_FORMATTED_TX_CSV)
    start_pos_df = load_start_positions(config.START_POSITIONS_CSV)

    # 2) Compute PnL curves
    initial_capital = 102_604  # or move this into config
    results_dict, uk_cum, uk_dd = aggregate_portfolio_pnl(
        tx=tx_df,
        prices=prices_df,
        start_pos_df=start_pos_df,
        initial_capital=config.INITIAL_SUB_PORTFOLIO_CAPITAL,
        start_date_str=config.START_DATE_STR
    )

    # 3) Now call plotting routines (in a separate module)
    # e.g.:
    # from agm.plotting import plot_individual, plot_aggregate
    # for name, (cum, dd) in results_dict.items():
    #     plot_individual(name, cum, dd, prices_df, uk_cum, uk_dd, output_folder)
    # plot_aggregate(results_dict, uk_cum, uk_dd, prices_df, output_folder)
    # ...
    print("PnL computation done; now generate charts.")
    return results_dict, uk_cum, uk_dd

if __name__ == "__main__":
    main()
