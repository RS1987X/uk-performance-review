from pathlib import Path
from datetime import date

# === USER‐EDITABLE SECTION ===
# Base folder under which everything lives
BASE_DIR = Path.home() / "Google Drive" / "UK AGM CMD"

# Name of the current AGM folder (update each year)
CURRENT_AGM_FOLDER = "AGM 20250531"

# Full path to the AGM directory
AGM_DIR = BASE_DIR / CURRENT_AGM_FOLDER

# Filenames (relative to AGM_DIR)
TRANSACTIONS_CSV = AGM_DIR / "transactions UK FKF.csv"
ISIN_MAPPING_CSV = AGM_DIR / "isin_to_ticker_map.csv"
FORMATTED_TRANSACTIONS_CSV = AGM_DIR / "formatted_transactions.csv"
PRICES_CSV = AGM_DIR / "closing_prices.csv"
START_POSITIONS_CSV = AGM_DIR / "starting_position.csv"
MANUAL_CSV_FOLDER = AGM_DIR / "missing data from yahoo"


OUTPUT_DIR = AGM_DIR / "charts"
# Dates
START_DATE_STR = "2024-08-16"
TODAY_STR = date.today().strftime("%Y-%m-%d")

INITIAL_PORTFOLIO_CAPITAL = 970_949
INITIAL_SUB_PORTFOLIO_CAPITAL = 102_604

SUB_PORTFOLIO_NAMES = ["SKUGG", "BERGMAN", "BURMAN", "GUNNARSSON", "JAEGERSTAD", "LINDHE", "SAFFAR", "SJOGREN", "SODERLUND", "STEFFEN"]


# API keys (if any)
EOD_API_KEY = "68397295a4b333.78618792"

# === END USER‐EDITABLE SECTION ===

# Derived configuration
HOME = Path.home()