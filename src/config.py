from pathlib import Path
from datetime import date

# === USER‐EDITABLE SECTION ===
# Base folder under which everything lives (now your GitHub clone)
BASE_DIR = Path.home() / "uk-agm-cmd-folder"

# Name of the current AGM folder (update each year)
# (if you no longer have a subfolder per year, just point these directly
#  at your CSVs, or remove them if not needed)
CURRENT_AGM_FOLDER = "AGM 20260601"

# Full path to the AGM directory
# (if your CSVs are now directly under BASE_DIR, you can drop this and use BASE_DIR directly)
AGM_DIR = BASE_DIR / CURRENT_AGM_FOLDER

# Filenames (relative to AGM_DIR)
SUBPORTF_PATH       = AGM_DIR / "transaction_subportfolios.csv"

AUGMENTED_TX_PATH = AGM_DIR / "augmented_transactions.csv"

RAW_TX_PATH = AGM_DIR / "transactions UK FKF.csv"

#TRANSACTIONS_CSV = AUGMENTED_TX_PATH#AGM_DIR / "transactions UK FKF.csv"

ISIN_MAPPING_CSV = AGM_DIR / "isin_to_ticker_map.csv"
#FORMATTED_TRANSACTIONS_CSV = AGM_DIR / "formatted_transactions.csv"
PRICES_CSV = AGM_DIR / "closing_prices.csv"
START_POSITIONS_CSV = AGM_DIR / "starting_position.csv"
MANUAL_CSV_FOLDER = AGM_DIR / "missing data from yahoo"

AUGMENTET_AND_FORMATTED_TX_CSV = AGM_DIR / "augmented_and_formatted_transactions.csv"

# If you prefer to keep everything at the top level of your repo (no YEAR subfolder),
# point them directly under BASE_DIR. For example:
# TRANSACTIONS_CSV = BASE_DIR / "transactions UK FKF.csv"
# PRICES_CSV       = BASE_DIR / "closing_prices.csv"
# etc.

# Where to write all output charts (this will now live under your repo)
OUTPUT_DIR = AGM_DIR / "charts"
# or, if you moved CSVs to the root of the repo:
# OUTPUT_DIR = BASE_DIR / "charts"

# Dates
START_DATE_STR = "2024-08-16"
TODAY_STR = date.today().strftime("%Y-%m-%d")
TODAY_STR = "2025-05-31"
# Initial capital values remain the same
INITIAL_PORTFOLIO_CAPITAL = 970_949
INITIAL_SUB_PORTFOLIO_CAPITAL = 102_604

SUB_PORTFOLIO_NAMES = [
    "SKUGG", "BERGMAN", "BURMAN", "GUNNARSSON",
    "JAEGERSTAD", "LINDHE", "SAFFAR", "SJOGREN",
    "SODERLUND", "STEFFEN"
]

# API keys (if any)
EOD_API_KEY = "68397295a4b333.78618792"

#GUI port
STREAMLIT_PORT = 8501
# === END USER‐EDITABLE SECTION ===

# Derived configuration
HOME = Path.home()