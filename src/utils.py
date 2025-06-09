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
# from pathlib import Path
# from typing import Union, List, Optional
# import pandas as pd
# import re
# from collections import Counter

# def read_flexible_table(
#     filepath: Union[str, Path],
#     encodings: Optional[List[str]] = None
# ) -> pd.DataFrame:
#     """
#     Read a delimited text table (transactions, prices, etc.) without assuming specific column names,
#     automatically handling:
#       1. Multiple encodings (utf-8, utf-8-sig, latin-1, cp1252).
#       2. Delimiters among [tab, semicolon, comma, pipe, runs of ≥2 spaces].
#          (If no single delimiter is consistent across sample lines, we fall
#          back to the regex (?:\t|;|,|\|)| {2,}.)
#       3. Decimal commas (',' → '.' in every field, then attempt to cast to float).
#       4. Malformed rows are skipped with a warning (on_bad_lines="warn").
#       5. The first non‐blank line is used as the header row.

#     Returns:
#       A pandas.DataFrame whose columns come exactly from the first non‐blank line,
#       with numeric columns converted to float (commas replaced by periods). If the
#       file is empty (or all‐blank), returns an empty DataFrame.
#     """
#     filepath = Path(filepath)
#     if encodings is None:
#         # Try these in order until we can at least load one non‐blank line
#         encodings = ["utf-16", "utf-8", "utf-8-sig", "latin-1", "cp1252"]

#     # 1) Try each encoding until we can read non‐blank lines into memory
#     last_exc = None
#     lines: List[str] = []
#     chosen_encoding: Optional[str] = None

#     for enc in encodings:
#         try:
#             with open(filepath, encoding=enc, errors="replace") as f:
#                 lines = [ln.rstrip("\n") for ln in f if ln.strip() != ""]
#             chosen_encoding = enc
#             break
#         except (UnicodeError, FileNotFoundError) as ue:
#             last_exc = ue
#             continue

#     if chosen_encoding is None:
#         raise last_exc or FileNotFoundError(f"Cannot open {filepath!r}")

#     if not lines:
#         # File is empty (or only blank lines)
#         return pd.DataFrame()

#     # 2) Sample up to N non‐blank lines for delimiter detection
#     sample_lines = lines[: min(len(lines), 10)]

#     # 3) Try to find a single “best” delimiter by checking consistency of column counts
#     #    among candidates: "\t", ";", ",", "|", or two (or more) spaces
#     candidates = [
#         ("\t", False),   # literal tab
#         (";", False),    # semicolon
#         (",", False),    # comma
#         ("|", False),    # pipe
#         (r" {2,}", True) # two or more spaces in a row ⇒ split on that
#     ]

#     best_sep: Optional[str] = None
#     best_score = -1
#     best_col_count = 0

#     for sep_pattern, is_regex in candidates:
#         counts = []
#         for ln in sample_lines:
#             text = ln.rstrip("\n")
#             if is_regex:
#                 parts = re.split(sep_pattern, text)
#             else:
#                 parts = text.split(sep_pattern)
#             counts.append(len(parts))

#         cnt = Counter(counts)
#         most_common_count, freq = cnt.most_common(1)[0]  # (column_count, number_of_lines_having_that_count)

#         # We only care if “most_common_count > 1” (it actually splits) and freq as large as possible.
#         if most_common_count > 1 and freq > best_score:
#             best_score = freq
#             best_col_count = most_common_count
#             best_sep = sep_pattern

#     # 4) If no single delimiter clearly “won,” fall back to the giant‐catchall regex:
#     #    (?:\t|;|,|\|)| {2,}
#     #    which means “either a tab, semicolon, comma, or pipe” OR “two (or more) spaces.”
#     if best_sep is None or best_score < 2:
#         unified_pattern = r"(?:\t|;|,|\|)| {2,}"
#         # Before committing, check that this unified actually splits into >1 field on at least one sample line:
#         unified_counts = [len(re.split(unified_pattern, ln.strip())) for ln in sample_lines]
#         if max(unified_counts) > 1:
#             sep_to_use = unified_pattern
#             sep_is_regex = True
#         else:
#             # In the very unlikely case that even the unified fails, fallback to comma:
#             sep_to_use = ","
#             sep_is_regex = False
#     else:
#         sep_to_use = best_sep
#         sep_is_regex = any(ch in best_sep for ch in r"\^$*+?{}[]|()")  # “is it a regex?”
#         # If best_sep was e.g. "\t" or ";" (no regex‐special chars), sep_is_regex=False

#     # 5) Now read with pandas, passing sep=sep_to_use (regex or literal)
#     read_kwargs = {
#         "encoding": chosen_encoding,
#         "engine": "python",
#         "on_bad_lines": "warn",
#     }

#     if sep_is_regex:
#         read_kwargs["sep"] = sep_to_use
#     else:
#         read_kwargs["sep"] = sep_to_use

#     df = pd.read_csv(filepath, **read_kwargs)

#     # 6) Post‐process every “object” column: replace ','→'.', strip whitespace, attempt numeric
#     for col in df.columns:
#         if df[col].dtype == object:
#             replaced = (
#                 df[col]
#                 .astype(str)
#                 .str.replace(",", ".", regex=False)  # decimal‐comma → decimal‐point
#                 .str.strip()
#             )
#             converted = pd.to_numeric(replaced, errors="ignore")
#             df[col] = converted

#     return df







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
        encodings = ["utf-16", "utf-8", "utf-8-sig", "latin-1", "cp1252"]

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

from typing import List, Optional, Union, Dict, Tuple
from pathlib import Path
import pandas as pd
import re

def detect_delimiter(header: str, candidate_delims: List[str] = ['\t', ';', ',', '|']) -> str:
    """Detect the most likely primary delimiter in the file."""
    counts = {delim: len(header.split(delim)) for delim in candidate_delims}
    return max(counts.items(), key=lambda x: x[1])[0]

def normalize_delimiters(
    input_path: Path,
    output_path: Path,
    encoding: str = 'utf-16',
    problem_columns: Optional[List[str]] = None,
    column_patterns: Optional[Dict[str, str]] = None
) -> None:
    """
    General purpose delimiter normalizer that:
    1. Auto-detects primary delimiter
    2. Handles problematic columns with custom splitting
    3. Normalizes all delimiters to tabs
    
    Args:
        input_path: Path to input file
        output_path: Path to write normalized file
        encoding: File encoding (default utf-16)
        problem_columns: List of column names that need special handling
        column_patterns: Dict mapping column names to regex patterns for splitting
    """
    # Read raw lines
    with open(input_path, 'r', encoding=encoding) as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        raise ValueError("Empty input file")
        
    # Detect primary delimiter from header
    header = lines[0]
    primary_delim = detect_delimiter(header)
    
    # Split header into columns
    header_cols = [col.strip() for col in header.split(primary_delim)]
    
    # Process data rows
    normalized_lines = [primary_delim.join(header_cols)]
    
    # Default pattern for cleaning up multiple spaces
    default_pattern = r'\s{2,}'
    
    for line in lines[1:]:
        # Split on primary delimiter first
        parts = [part.strip() for part in line.split(primary_delim)]
        
        # Handle problem columns if specified
        if problem_columns and column_patterns:
            for col_name, pattern in column_patterns.items():
                try:
                    col_idx = header_cols.index(col_name)
                    if col_idx < len(parts):
                        # Split the problematic column using its pattern
                        split_values = [
                            v.strip() for v in 
                            re.split(pattern, parts[col_idx]) 
                            if v.strip()
                        ]
                        parts[col_idx:col_idx+1] = split_values
                except ValueError:
                    continue  # Column not found in header
        
        # Join with primary delimiter
        normalized_lines.append(primary_delim.join(parts))
    
    # Write normalized file
    with open(output_path, 'w', encoding=encoding) as f:
        f.write('\n'.join(normalized_lines))

def normalize_transactions(input_path: Path, output_path: Path) -> None:
    """
    Specific normalizer for transaction files that handles both:
    1. Depå + Portfolio spacing issues
    2. Portfolio + Transaktionstyp spacing issues
    """
    # Define problematic column names and their splitting patterns
    problem_cols = [
        "Depå    Portfolio   Transaktionstyp",  # Combined column
        "Depå    Portfolio",                    # In case partially combined
        "Portfolio   Transaktionstyp"           # In case partially combined
    ]
    
    patterns = {
        "Depå    Portfolio   Transaktionstyp": r'\s{2,}',  # Split on 2+ spaces
        "Depå    Portfolio": r'\s{2,}',
        "Portfolio   Transaktionstyp": r'\s{2,}'
    }
    
    normalize_delimiters(
        input_path=input_path,
        output_path=output_path,
        encoding='utf-16',
        problem_columns=problem_cols,
        column_patterns=patterns
    )