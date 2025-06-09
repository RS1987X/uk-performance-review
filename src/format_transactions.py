# agm/format_transactions.py

import pandas as pd
from pathlib import Path
from src import config
from src.utils import read_messy_tab_file,read_flexible_table,normalize_transactions
from typing import Optional


def format_transactions(
    augmented_transactions_path: Path,
    mapping_path: Path,
    output_path: Path
) -> pd.DataFrame:
    """
    - Load transactions using the “messy” reader.
    - Merge with an ISIN→ticker mapping.
    - Create columns: ’Name YFINANCE’, ’Belopp SEK (ex courtage)’, ’Pris SEK’.
    - Reorder to the desired columns and save as a semi‐colon‐delimited CSV.
    Returns the formatted DataFrame.
    """
    # temp_path = raw_transactions_path.with_suffix('.tmp')
    # normalize_transactions(raw_transactions_path, temp_path)

    # 1. Read raw transactions (could be tab / messy)
    #tx = read_messy_tab_file(raw_transactions_path)
    #tx = read_flexible_table(raw_transactions_path)
    #tx = read_flexible_table(augmented_transactions_path)
    #tx = read_messy_tab_file(augmented_transactions_path, encoding="utf-8").values
    tx = pd.read_csv(augmented_transactions_path, sep=",", encoding="utf-8")
    #for the column "Kurs" change the decimal separator from "," to "."
    #tx["Kurs"] = tx["Kurs"].str.replace(",", ".", regex=False)
    #tx = tx.str.replace(",", ".", regex=False)
    cols_to_replace = ["Kurs", "Belopp", "Antal"]
    for col in cols_to_replace:
        tx[col] = tx[col].astype(str).str.replace(",", ".", regex=False)
    # 2. Check that 'Portfolio' has been created by the user
    if "subportfolio" not in tx.columns:
        raise RuntimeError(
            f"❌  'subortfolio' column not found in {augmented_transactions_path}. "
            f"Please add a 'subportfolio' column before running this script."
        )


    # 2. Read mapping
    isin_map = pd.read_csv(mapping_path)
    # Fill missing or empty strings with NA
    isin_map["TICKER"] = isin_map["TICKER"].replace("", pd.NA)

    # 3. Merge to get the YFinance ticker
    merged = tx.merge(isin_map, on="ISIN", how="left")

    # 4. Apply any manual overrides from data/ticker_overrides.csv
    overrides_path = Path(__file__).parent.parent / "data" / "ticker_overrides.csv"
    if overrides_path.exists():
        ov_df = pd.read_csv(overrides_path, dtype=str)
        ov_df.columns = [c.strip() for c in ov_df.columns]
        # Build a dict { ISIN → OVERRIDE_TICKER }
        override_map = {
            str(r["ISIN"]).strip(): str(r["TICKER"]).strip()
            for _, r in ov_df.iterrows()
            if pd.notna(r.get("ISIN")) and pd.notna(r.get("TICKER"))
        }
        # Wherever a transaction’s ISIN is in override_map, force the new TICKER
        # (ignoring whatever came from the original isin_map)
        mask = merged["ISIN"].isin(override_map.keys())
        if mask.any():
            merged.loc[mask, "TICKER"] = merged.loc[mask, "ISIN"].map(override_map)
    else:
        # If there's no overrides file, just proceed
        override_map = {}


    # 4. Replace NaN → "" so that we can apply string operations safely
    merged["TICKER"] = merged["TICKER"].fillna("")

    # 5. Build “Name YFINANCE”: use TICKER only if it’s non‐blank AND
    #    does not start with “No match” or “Error” (case‐sensitive).
    invalid_mask = (
        (merged["TICKER"] == "") |
        merged["TICKER"].str.startswith("No match") |
        merged["TICKER"].str.startswith("Error")
    )
    merged["Name YFINANCE"] = merged["TICKER"].where(
        ~invalid_mask,
        merged["Värdepapper"]
        .str.replace(" ", "-", regex=False)
    )

    # Remove trailing ".US"
    merged["Name YFINANCE"] = merged["Name YFINANCE"].str.replace(r"\.US$", "", regex=True)
    
    #rename columns for clarity
    merged.rename(
    columns={"Name YFINANCE": "Identifying name"},
    inplace=True
)

    # 4. Convert numeric columns (Belopp, Antal) to numeric dtypes
    # for col in ("Belopp", "Antal"):
    #     merged[col] = merged[col].astype(str)
    #     merged["Belopp"] = merged["Belopp"].str.replace(",", ".", regex=False)
    #     merged["Antal"] = (
    #         merged["Antal"]
    #         .str.replace(r"\.", "", regex=True)    # remove thousands separator
    #         .str.replace(",", ".", regex=False)
    #     )
    merged["Belopp"] = pd.to_numeric(merged["Belopp"], errors="coerce")
    merged["Antal"] = pd.to_numeric(merged["Antal"], errors="coerce")

    # 5. Compute new fields
    merged["Belopp SEK (ex courtage)"] = merged["Belopp"]  # could rename if needed
    merged["Pris SEK"] = merged["Belopp SEK (ex courtage)"] / merged["Antal"]

    # 6. Select & reorder columns
    output_cols = [
        "Affärsdag",
        "subportfolio",
        "Identifying name",
        "Transaktionstyp",
        "Antal",
        "Belopp SEK (ex courtage)",
        "Pris SEK",
        "Valuta.3"
    ]
    output_df = merged[output_cols]

    # 7. Save with semi‐colon separator
    output_df.to_csv(output_path, sep=";", index=False)
    print(f"✔ Saved transformed CSV to {output_path}")
    return output_df


def main():
    df = format_transactions(
        augmented_transactions_path=config.AUGMENTED_TX_PATH,
        mapping_path=config.ISIN_MAPPING_CSV,
        output_path=config.AUGMENTET_AND_FORMATTED_TX_CSV,
    )


if __name__ == "__main__":
    main()
