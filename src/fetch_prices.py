# agm/fetch_prices.py

import os
import pandas as pd
import yfinance as yf
from pathlib import Path
from src import config
from typing import Dict, Optional
import logging
from src.utils import read_flexible_table

# === Setup logging ===
log_file = config.AGM_DIR / "fetch_prices.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

def fetch_ticker_prices(
    mapping_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    manual_data_dir: Path,
    starting_positions_csv: Optional[Path] = None
) -> Dict[str, pd.Series]:
    """
    Given a DataFrame with columns ["ISIN", "TICKER"], fetch historical closing prices for each ticker:
      1. First, apply any overrides in AGM_DIR/ticker_overrides.csv.
      2. Then, for each ISIN/ticker pair:
         a) If ticker is valid, try yfinance.
         b) If yfinance returns empty or errors, or if ticker was invalid,
            attempt to load manual CSV from manual_data_dir/<ISIN>.csv.
         c) If both fail, record a single failure for this ISIN.

    Returns a dict mapping (ticker_or_isin → pd.Series of closing prices).
    Also writes price_fetch_errors.csv under AGM_DIR if any failures occur.
    """
    price_series: Dict[str, pd.Series] = {}
    price_failures = []  # Will collect {"ISIN":…, "Ticker":…, "Reason":…} entries

    # --- (A)  Insert starting‐positions tickers into mapping_df ---
    if starting_positions_csv is not None and starting_positions_csv.exists():
        sp_df = read_flexible_table(starting_positions_csv)
        # We expect a column named "Name" in sp_df that already contains tickers like "AMZN", "BRE2.ST", etc.
        if "Name" not in sp_df.columns:
            raise ValueError(f"Expected a 'Name' column in {starting_positions_csv}, but got: {list(sp_df.columns)}")

        # Extract unique tickers (non‐empty, stripped)
        sp_tickers = sorted(
            { str(t).strip() for t in sp_df["Name"].dropna().tolist() if str(t).strip() != "" }
        )
        # Build a mini‐DataFrame where ISIN=TICKER=<that string>
        # But only add if that ticker isn't already in the mapping_df["TICKER"] set
        #existing_tickers = { str(t).strip() for t in mapping_df["TICKER"].dropna().tolist() }
        existing_values = {
            str(x).strip()
            for col in ("TICKER", "ISIN")
            for x in mapping_df[col].dropna()
        }
                
        rows_to_add = []
        for tk in sp_tickers:
            if tk not in existing_values:
                rows_to_add.append({"ISIN": tk, "TICKER": tk})
                logger.info(f"Adding starting‐position ticker to mapping: ISIN/TICKER = '{tk}'")
        if rows_to_add:
            extra_df = pd.DataFrame(rows_to_add)
            # Prepend (or append) to mapping_df—it doesn't really matter, but prepending makes it clear these came from SP.
            mapping_df = pd.concat([extra_df, mapping_df], ignore_index=True)
        else:
            logger.info("No new starting‐position tickers to add (all already in mapping).")



    # --- 1) Load and apply overrides from AGM_DIR/ticker_overrides.csv, if present ---
    overrides_path = overrides_path = Path(__file__).parent.parent / "data" / "ticker_overrides.csv"
    if overrides_path.exists():
        ov = pd.read_csv(overrides_path, dtype=str)
        ov.columns = [c.strip() for c in ov.columns]
        override_map = {
            str(r["ISIN"]).strip(): str(r["TICKER"]).strip()
            for _, r in ov.iterrows()
            if pd.notna(r.get("ISIN")) and pd.notna(r.get("TICKER"))
        }
        for idx, row in mapping_df.iterrows():
            isin = str(row["ISIN"]).strip()
            if isin in override_map:
                old_t = mapping_df.at[idx, "TICKER"]
                new_t = override_map[isin]
                mapping_df.at[idx, "TICKER"] = new_t
                logger.info(f"OVERRIDE: ISIN {isin!r}: {old_t!r} → {new_t!r}")
    else:
        override_map = {}

    # --- 2) Loop over each ISIN/ticker, try Yahoo then manual fallback ---
    for _, row in mapping_df.iterrows():
        isin = row["ISIN"]
        ticker = row["TICKER"]

        # a) Clean up trailing ".US" if present
        if isinstance(ticker, str) and ticker.endswith(".US"):
            ticker = ticker[:-3]
        if ticker == "^OMXSGI":
            print("da")
        # b) Try yfinance only if ticker is not empty and doesn’t start with “No match” or “Error”
        got_from_yahoo = False
        if pd.notna(ticker) and not ticker.startswith(("No match", "Error")):
            logger.info(f"Fetching data for ISIN {isin!r} → ticker {ticker!r}")
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    closing = data["Close"]
                    closing.name = ticker
                    closing.index = pd.to_datetime(closing.index)
                    price_series[ticker] = closing
                    logger.info(f"  ✓ Retrieved {len(closing)} rows for {ticker!r}")
                    got_from_yahoo = True
                else:
                    reason = "Empty yfinance data"
                    logger.warning(f"  ⚠️ No data found on Yahoo for {ticker!r} (ISIN={isin!r})")
                    # fall through to manual fallback
            except Exception as e:
                reason = f"Exception while fetching from Yahoo: {e}"
                logger.error(f"  ❌ {reason} for {ticker!r} (ISIN={isin!r})")
                # fall through to manual fallback
        else:
            # Mark as invalid and fall through to manual fallback
            reason = "Invalid ticker or mapping"
            logger.warning(f"{reason} for ISIN {isin!r}: {ticker!r}")

        # c) If Yahoo succeeded, skip manual fallback
        if got_from_yahoo:
            continue
        
        
        # d) Manual fallback (runs if Yahoo failed OR ticker was invalid)
        file_path = manual_data_dir / f"{ticker}.csv"
        if file_path.exists():
            # if isin == "IE00B4M7GH52":
            #     print("da")
            logger.info(f"Attempting manual fallback for Identifying name={ticker!r} from {file_path}")
            try:
                # df_manual = pd.read_csv(
                #     file_path,
                #     sep="\t",
                #     decimal=",",
                #     encoding="utf-8",
                #     header=None,
                #     skiprows=1
                # )
                # df_manual = pd.read_csv(
                #     file_path,
                #     sep="\t",
                #     decimal=",",
                #     encoding="utf-8"
                #     # no header=None or skiprows—let pandas read its first row as the header
                # )
                df_manual = read_flexible_table(file_path)
                 # 2) Verify that there is a “Date” and a “Last” column.
                if "Date" not in df_manual.columns or "Last" not in df_manual.columns:
                    raise ValueError(f"Expected columns 'Date' and 'Last' not found. Got: {list(df_manual.columns)}")

                 # 3) Convert “Date” to datetime and set it as the index.
                df_manual["Date"] = pd.to_datetime(df_manual["Date"], errors="coerce")
                df_manual.set_index("Date", inplace=True)

                # 4) Extract the “Last” column as our closing prices.
                closing = df_manual["Last"].rename(ticker)
                price_series[ticker] = closing
                logger.info(f"  ✓ Loaded {len(closing)} rows of manual data for Identifying name={ticker!r}")
                continue
            except Exception as e:
                reason = f"Error reading manual CSV: {e}"
                logger.error(f"  ❌ {reason} for Identifying name={ticker!r} at path {file_path}")
                price_failures.append({
                    "ISIN": isin,
                    "Ticker": ticker,
                    "Reason": reason
                })
                continue  # Move to next ISIN even though manual fallback failed

        # e) If we reach here, neither Yahoo nor Manual succeeded for this ISIN
        final_reason = reason if not got_from_yahoo else "No Yahoo data AND no manual CSV"
        logger.warning(
            f"⚠️ No data found for ISIN={isin!r} (ticker={ticker!r}); "
            f"manual file {file_path} not found"
        )
        price_failures.append({
            "ISIN": isin,
            "Ticker": ticker,
            "Reason": final_reason
        })

    # --- 3) After the loop, save all failures (if any) to a CSV ---
    if price_failures:
        pf_df = pd.DataFrame(price_failures)
        pf_path = config.AGM_DIR / "price_fetch_errors.csv"
        pf_df.to_csv(pf_path, index=False, encoding="utf-8")
        logger.info(f"Wrote {len(pf_df)} fetch‐error rows to {pf_path}")
    else:
        logger.info("All tickers fetched successfully—no price_fetch_errors.csv generated.")

    return price_series



def fetch_fx_rates(
    formatted_tx_path: Path,
    start_date: str,
    end_date: str
) -> Dict[str, pd.Series]:
    """
    Read the formatted transactions CSV, gather all unique non‐SEK currencies
    (from any column starting with 'Valuta'), and fetch fx rates (CCYSEK=X) from yfinance.
    Returns a dict mapping 'CCYSEK=X' → pd.Series of Close prices.
    """
    fx_series: Dict[str, pd.Series] = {}
    if not formatted_tx_path.exists():
        print("⚠️ formatted_transactions.csv not found — skipping FX rate collection.")
        return fx_series

    fx_df = pd.read_csv(formatted_tx_path, sep=";", encoding="utf-8")
    valuta_cols = [col for col in fx_df.columns if col.startswith("Valuta")]
    raw_ccys = pd.unique(fx_df[valuta_cols].values.ravel())
    ccy_list = sorted({ccy.strip() for ccy in raw_ccys if isinstance(ccy, str) and ccy.strip() and ccy.strip() != "SEK"})
    print(f"\n🌍 Currencies to fetch: {ccy_list}")

    for ccy in ccy_list:
        fx_ticker = f"{ccy}SEK=X"
        print(f"Fetching FX rate for {ccy} → {fx_ticker}")
        try:
            fx_data = yf.download(fx_ticker, start=start_date, end=end_date, progress=False)
            if not fx_data.empty:
                fx_close = fx_data["Close"]
                fx_close.name = fx_ticker
                fx_close.index = pd.to_datetime(fx_close.index)
                fx_series[fx_ticker] = fx_close
            else:
                print(f"⚠️ No FX data found for {fx_ticker}")
        except Exception as e:
            print(f"❌ Error fetching FX rate for {ccy}: {e}")

    return fx_series


def fetch_omx(index_symbol: str, start_date: str, end_date: str) -> pd.Series:
    """
    Always attempt to fetch the OMXSGI index (or any given index_symbol).
    Returns a pd.Series of Close prices (named by index_symbol).
    """
    print("\n📈 Fetching OMXSGI index price")
    try:
        omx_data = yf.download(index_symbol, start=start_date, end=end_date, progress=False)
        if not omx_data.empty:
            omx_close = omx_data["Close"]
            omx_close.name = index_symbol
            omx_close.index = pd.to_datetime(omx_close.index)
            return omx_close
        else:
            print(f"⚠️ No data found for {index_symbol}")
            return pd.Series(name=index_symbol, dtype=float)
    except Exception as e:
        print(f"❌ Error fetching {index_symbol}: {e}")
        return pd.Series(name=index_symbol, dtype=float)


def combine_and_save(
    price_series_dict: Dict[str, pd.Series],
    fx_series_dict: Dict[str, pd.Series],
    omx_series: pd.Series,
    output_path: Path
) -> pd.DataFrame:
    """
    Concatenate all series (tickers, FX, and OMX) along the union of dates (outer join).
    Write to output_path as CSV. Returns the combined DataFrame.
    """
    all_series = {**price_series_dict, **fx_series_dict}
    if not omx_series.empty:
        all_series[omx_series.name] = omx_series

    if all_series:
        combined_df = pd.concat(all_series.values(), axis=1).sort_index()
        combined_df.to_csv(output_path)
        print(f"\n✅ Combined closing prices saved to {output_path}")
        return combined_df
    else:
        print("❌ No valid data to combine. Nothing saved.")
        return pd.DataFrame()


def main():
    # 1. Load ISIN→TICKER
    mapping_df = pd.read_csv(config.ISIN_MAPPING_CSV)

    # 1a. Inject the OMXSGI index as “just another ticker”:
    #     Only add it if it doesn’t already appear in either column.
    omx_symbol = "^OMXSGI"
    omx_isin = "SE0002416156"
    existing_values = {
        str(v).strip()
        for col in ("ISIN", "TICKER")
        for v in mapping_df[col].dropna().tolist()
    }
    if omx_symbol not in existing_values:
        # We prepend so that logging/fetch‐order shows the index first.
        omx_row = pd.DataFrame([{"ISIN": omx_isin, "TICKER": omx_symbol}])
        mapping_df = pd.concat([omx_row, mapping_df], ignore_index=True)

    # 2. Fetch prices
    price_series = fetch_ticker_prices(
        mapping_df=mapping_df,
        start_date=config.START_DATE_STR,
        end_date=config.TODAY_STR,
        manual_data_dir=config.MANUAL_CSV_FOLDER,
        starting_positions_csv=config.START_POSITIONS_CSV
    )

    # 3. Fetch FX rates
    fx_series = fetch_fx_rates(
        formatted_tx_path=config.AUGMENTET_AND_FORMATTED_TX_CSV,
        start_date=config.START_DATE_STR,
        end_date=config.TODAY_STR
    )

    # 4. Fetch OMXSGI
    #omx_series = fetch_omx("^OMXSGI", config.START_DATE_STR, config.TODAY_STR)
    omx_series = price_series.pop(omx_isin, pd.Series(name=omx_isin, dtype=float))

    # 5. Combine & save
    _ = combine_and_save(
        price_series_dict=price_series,
        fx_series_dict=fx_series,
        omx_series=omx_series,
        output_path=config.PRICES_CSV
    )


if __name__ == "__main__":
    main()
