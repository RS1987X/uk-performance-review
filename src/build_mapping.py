# agm/build_mapping.py

import requests
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from src import config
from src.utils import read_messy_tab_file
import logging

# === Setup logging ===
log_file = config.AGM_DIR / "mapping.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),             # Console
        logging.FileHandler(log_file, mode="a", encoding="utf-8")  # Append to mapping.log
    ]
)
logger = logging.getLogger(__name__)


def isin_to_ticker(isin: str, api_key: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Query EOD Historical Data API for a given ISIN and return (ticker_code, exchange).
    If no match, returns (None, reason_string).
    """
    url = f"https://eodhistoricaldata.com/api/search/{isin}?api_token={api_key}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return None, f"Error: {resp.status_code}"
    data = resp.json()
    if not data:
        return None, "No match"
    # Take the first match
    return data[0]["Code"], data[0]["Exchange"]


# def build_mapping(
#     transactions_csv: Path,
#     output_csv: Path,
#     api_key: str
# ) -> pd.DataFrame:
#     """
#     Reads the transactions file, extracts all unique ISINs, queries EOD,
#     and writes a CSV mapping ISIN → ticker (with exchange suffix).
#     Returns the DataFrame for further use if needed.
#     """
#     df = read_messy_tab_file(transactions_csv)
#     unique_isins = df["ISIN"].dropna().unique().tolist()

#     records = []
#     failures = []
#     for isin in unique_isins:
#         code, exchange = isin_to_ticker(isin, api_key)
#         if code:
#             code_clean = code.replace(" ", "-")
#             ticker_str = f"{code_clean}.{exchange}"
#             records.append({"ISIN": isin, "TICKER": ticker_str})
#             logger.info(f"Mapped ISIN {isin!r} → {ticker_str!r}")
#         else:
#             ticker_str = exchange  # holds the “reason”
#             failures.append({"ISIN": isin, "Reason": exchange})
        

#     mapping_df = pd.DataFrame(records)
#     mapping_df.to_csv(output_csv, index=False, encoding="utf-8")
#     logger.info(f"Wrote mapping file to {output_csv} ({len(mapping_df)} entries)")

#     # Save any failures to a CSV
#     if failures:
#         failures_df = pd.DataFrame(failures)
#         fail_path = config.AGM_DIR / "mapping_failures.csv"
#         failures_df.to_csv(fail_path, index=False, encoding="utf-8")
#         logger.info(f"{len(failures_df)} ISINs failed. See {fail_path}")
#     else:
#         logger.info("No mapping failures detected.")

#     return mapping_df


def build_mapping(
    transactions_csv: Path,
    output_csv: Path,
    api_key: str
) -> pd.DataFrame:
    """
    1) Load any existing mapping CSV (output_csv) into a dict.
    2) Read the transactions file to find all unique ISINs.
    3) For each ISIN not already in the existing mapping, call EOD once.
    4) Merge “old” + “new” mappings into one DataFrame, write to output_csv.
    5) If any lookups failed, write them to mapping_failures.csv.
    """
    # --- 1. Load existing mapping (if it exists) ---
    if output_csv.exists():
        existing_df = pd.read_csv(output_csv, dtype=str)
        # Normalize column names (in case of stray whitespace)
        existing_df.columns = [c.strip() for c in existing_df.columns]
        # Build a dict: { ISIN → TICKER }
        existing_map: Dict[str, str] = {
            str(row["ISIN"]).strip(): str(row["TICKER"]).strip()
            for _, row in existing_df.iterrows()
            if pd.notna(row["ISIN"]) and pd.notna(row["TICKER"])
        }
        logger.info(
            f"Loaded {len(existing_map)} existing mappings from {output_csv}"
        )
    else:
        existing_map = {}
        logger.info(f"No existing mapping file found at {output_csv}; starting fresh.")

    # --- 2. Read raw transactions & find unique ISINs ---
    df = read_messy_tab_file(transactions_csv)
    unique_isins = [str(i).strip() for i in df["ISIN"].dropna().unique().tolist()]

    # Prepare containers for any newly‐discovered mappings or failures
    new_records = []
    failures = []
    
    # --- 3. For each ISIN not in existing_map, call EOD ---
    for isin in unique_isins:
        if isin in existing_map:
            # We already have this ISIN→ticker; skip the API call
            logger.debug(f"ISIN {isin!r} already mapped to {existing_map[isin]!r}; skipping EOD.")
            continue
        elif isin == '':
            logger.debug(f"ISIN {isin!r} is empty string; skipping EOD.")
            continue
        
        logger.info(f"Querying EOD for ISIN {isin!r}")
        code, exchange = isin_to_ticker(isin, api_key)

        if code:
            code_clean = code.replace(" ", "-")
            ticker_str = f"{code_clean}.{exchange}"
            new_records.append({"ISIN": isin, "TICKER": ticker_str})
            logger.info(f"  ✓ Got mapping {isin!r} → {ticker_str!r}")
        else:
            # Record failure (so you can inspect it)
            reason = exchange or "Unknown reason"
            failures.append({"ISIN": isin, "Reason": reason})
            logger.warning(f"  ⚠️  No mapping for ISIN {isin!r}: {reason!r}")

    # --- 4. Combine “old” + “new” into one DataFrame ---
    #    a) Turn existing_map dict → DataFrame
    existing_list = [
        {"ISIN": isin_key, "TICKER": existing_map[isin_key]}
        for isin_key in existing_map
    ]
    combined_df = pd.DataFrame(existing_list + new_records)

    #    b) Write the updated mapping back to output_csv
    combined_df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info(f"Wrote {len(combined_df)} ISIN→ticker records to {output_csv}")

    # --- 5. If there were any failures, write them out separately ---
    if failures:
        fail_df = pd.DataFrame(failures)
        fail_path = output_csv.parent / "mapping_failures.csv"
        fail_df.to_csv(fail_path, index=False, encoding="utf-8")
        logger.info(f"{len(fail_df)} ISINs failed to map; see {fail_path}")
    else:
        logger.info("No new mapping failures detected.")

    return combined_df


def main():
    mapping_df = build_mapping(
        transactions_csv=config.RAW_TX_PATH,
        output_csv=config.ISIN_MAPPING_CSV,
        api_key=config.EOD_API_KEY,
    )
    print(f"✅ Wrote {len(mapping_df)} ISIN→ticker mappings to {config.ISIN_MAPPING_CSV}")


if __name__ == "__main__":
    main()
