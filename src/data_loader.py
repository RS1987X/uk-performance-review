# agm/data_loader.py

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
from typing import Dict, Tuple
from src import config


def load_prices(prices_csv: Path) -> pd.DataFrame:
    """
    Read 'closing_prices.csv' and return a DataFrame indexed by date (as date objects),
    with columns for each ticker or FX pair.
    Replace zeros with NaN and forward‐fill.
    """
    prices = pd.read_csv(prices_csv, parse_dates=["Date"])
    prices.set_index("Date", inplace=True)
    prices.index = prices.index.date
    # Replace 0.0 with NaN; forward‐fill
    prices = prices.replace(0, np.nan).ffill()
    return prices


def load_transactions(formatted_tx_csv: Path) -> pd.DataFrame:
    """
    Read the formatted transactions CSV, parse dates, normalize column names, etc.
    """
    tx = pd.read_csv(formatted_tx_csv, sep=";", parse_dates=["Affärsdag"], dayfirst=True)
    tx.rename(columns={"Affärsdag": "Date"}, inplace=True)
    tx["subportfolio"] = tx["subportfolio"].fillna("").astype(str).str.upper()
    tx["Date"] = tx["Date"].dt.date
    return tx


def load_start_positions(start_pos_csv: Path) -> pd.DataFrame:
    """
    If the starting positions CSV exists, load it; otherwise return an empty DataFrame.
    Ensure 'Portfolio' is uppercase.
    """
    if start_pos_csv.exists():
        start_df = pd.read_csv(start_pos_csv, sep=";")
        start_df["subportfolio"] = start_df["subportfolio"].astype(str).str.upper()
        # Filter out blank portfolio lines
        return start_df[start_df["subportfolio"] != ""]
    else:
        return pd.DataFrame(columns=["subportfolio", "Name", "Shares", "ccy"])


def infer_currency_map(
    tx: pd.DataFrame,
    start_pos_df: pd.DataFrame
) -> Dict[str, str]:
    """
    Build a dictionary { ticker_str: currency_str } by scanning the 'Valuta' columns
    in tx and any entries in start_pos_df. Duplicate‐currency warnings are printed.
    """
    valuta_cols = [c for c in tx.columns if c.startswith("Valuta")]
    currency_map = {}

    # 1) From buy/sell rows in tx
    for name, group in tx.groupby("Name YFINANCE"):
        trades_bs = group[group["Transaktionstyp"].isin(["KÖPT", "SÅLT"])]
        if trades_bs.empty:
            # no PnL trades; assume SEK unless overwritten by start_pos
            continue
        vals = trades_bs[valuta_cols].values.ravel()
        cands = [v for v in vals if isinstance(v, str) and v not in ("", "SEK")]
        unique_ccys = set(cands)
        if not unique_ccys:
            ccy = "SEK"
        elif len(unique_ccys) == 1:
            ccy = unique_ccys.pop()
        else:
            ccy = sorted(unique_ccys)[0]
            print(f"⚠️ {name} appears in multiple currencies {unique_ccys}, defaulting to {ccy}")
        currency_map[name] = ccy

    # 2) Overlay any missing tickers from starting positions
    for _, row in start_pos_df.iterrows():
        ticker = row["Name"]
        ccy = row.get("ccy", None)
        if not ccy or str(ccy).strip() == "":
            print(f"⚠️ No CCY found for {ticker} in starting_position.csv; assuming SEK")
            ccy = "SEK"
        if ticker not in currency_map:
            currency_map[ticker] = ccy

    # 3) Report any tickers that remain missing
    tx_tickers = set(tx["Name YFINANCE"].dropna().unique())
    start_tickers = set(start_pos_df["Name"].dropna().unique())
    all_tickers = tx_tickers.union(start_tickers)
    missing = sorted([t for t in all_tickers if t not in currency_map])
    if missing:
        print("Tickers present in tx or starting positions but missing from currency_map:")
        for t in missing:
            print(f"  • {t}")
    else:
        print("All tickers from tx and starting positions are in currency_map.")
    return currency_map


def load_omx_manual(omx_manual_csv: Path) -> pd.Series:
    """
    If you need to read an OMXSGI from a manual file (instead of yfinance),
    implement that logic here. Otherwise, let plotter read from prices directly.
    """
    # (In the refactored flow, we always fetch OMX via yfinance, so this may be unnecessary.)
    raise NotImplementedError("Manual OMX loader not implemented.")


def main():
    # Example usage if run directly
    prices = load_prices(config.PRICES_CSV)
    tx = load_transactions(config.FORMATTED_TRANSACTIONS_CSV)
    start_pos = load_start_positions(config.START_POSITIONS_CSV)
    currency_map = infer_currency_map(tx, start_pos)
    # Usually you’d return these to callers rather than printing.


if __name__ == "__main__":
    main()
