from collections import Counter
import pandas as pd

def read_messy_tab_file(filepath, encoding='utf-16', preview_bad_rows=5):
    """
    Reads a tab-delimited file with possibly inconsistent row lengths
    and returns a cleaned pandas DataFrame. Also prints diagnostics on malformed rows.
    
    Preserves empty trailing fields by only stripping newline, not all whitespace.
    """
    rows = []
    with open(filepath, encoding=encoding) as f:
        for line_num, line in enumerate(f, 1):
            # Only remove the trailing newline, not tabs or spaces
            fields = line.rstrip('\n').split('\t')
            rows.append(fields)

    # Determine the most common row length
    lengths = Counter(len(row) for row in rows)
    most_common_length = lengths.most_common(1)[0][0]
    print(f"Detected most common row length: {most_common_length}")
    print(f"Row length distribution: {dict(lengths)}")

    # Identify and print malformed rows
    bad_rows = [(i + 1, row) for i, row in enumerate(rows) if len(row) != most_common_length]
    if bad_rows:
        print(f"\n⚠️ Found {len(bad_rows)} malformed rows (length != {most_common_length}).")
        for i, (line_num, row) in enumerate(bad_rows):
            if i >= preview_bad_rows:
                print(f"...and {len(bad_rows) - preview_bad_rows} more.")
                break
            print(f"Line {line_num}: {len(row)} fields -> {row}")
    else:
        print("✅ All rows match the expected format.")

    # Normalize rows
    cleaned_rows = []
    for row in rows:
        if len(row) < most_common_length:
            row += [''] * (most_common_length - len(row))
        elif len(row) > most_common_length:
            row = row[:most_common_length]
        cleaned_rows.append(row)

    # Convert to DataFrame
    header = cleaned_rows[0]
    data = pd.DataFrame(cleaned_rows[1:], columns=header)
    return data
# src/utils.py

import pandas as pd
from pathlib import Path
from typing import Union, Optional, List


def read_flexible_table(
    filepath: Union[str, Path],
    encodings: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Read a delimited text table (transactions, prices, etc.) without assuming specific column names,
    automatically handling:
      1. Multiple encodings (utf-8, utf-8-sig, latin-1, cp1252).
      2. Delimiters among [tab, semicolon, comma, pipe, whitespace].
      3. Decimal commas (it replaces ','→'.' in every field and attempts to cast to float).
      4. Malformed rows are skipped with a warning (on_bad_lines="warn").
      5. The first non‐blank line is used as the header row.

    Returns:
      A pandas.DataFrame whose columns come exactly from the first non‐blank line, with numeric
      columns converted to float (commas replaced by periods).
    """
    filepath = Path(filepath)
    if encodings is None:
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    # 1) Try each encoding until we can read all non-blank lines into memory:
    last_exc = None
    lines = []
    chosen_encoding = None

    for enc in encodings:
        try:
            with open(filepath, encoding=enc, errors="replace") as f:
                # Keep only non-blank lines (rstrip newline, but preserve spaces/tabs inside)
                lines = [line.rstrip("\n") for line in f if line.strip() != ""]
            chosen_encoding = enc
            break
        except UnicodeError as ue:
            last_exc = ue
            continue

    if chosen_encoding is None:
        raise last_exc or FileNotFoundError(f"Cannot open {filepath!r}")

    if not lines:
        # Empty or all-blank file
        return pd.DataFrame()

    # 2) The first non-blank line is the header:
    header_line = lines[0]

    # 3) Choose a delimiter by testing each candidate until it splits header into >1 field
    sep = None
    for candidate in ["\t", ";", ",", "|", " "]:
        if candidate == " ":
            parts = header_line.split()
        else:
            parts = header_line.split(candidate)
        if len(parts) > 1:
            sep = candidate
            break
    if sep is None:
        sep = ","  # fallback

    # 4) Read into a DataFrame via pandas with that delimiter & chosen encoding
    if sep == " ":
        df = pd.read_csv(
            filepath,
            sep=r'\s+',
            encoding=chosen_encoding,
            engine="python",
            on_bad_lines="warn"
        )
    else:
        df = pd.read_csv(
            filepath,
            sep=sep,
            encoding=chosen_encoding,
            engine="python",
            on_bad_lines="warn"
        )

    # 5) Normalize decimal commas → periods and convert numeric columns
    for col in df.columns:
        if df[col].dtype == object:
            # Replace every comma with a period
            replaced = df[col].str.replace(",", ".", regex=False).str.strip()
            # Attempt to convert to numeric. If it fails, keep as-is.
            converted = pd.to_numeric(replaced, errors="ignore")
            df[col] = converted

    return df
