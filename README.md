# UK AGM Performance Review

This project computes portfolio, sub‐portfolio and security PnL, compares it versus OMXSGI, and generates performance tables and charts. Below are the steps to follow each year (or on any new machine) to update data and run the analysis.

---

## 1. Prerequisites

- **WSL installed** (Ubuntu or your preferred distro)
- **Python 3.8+** available in WSL
- **Git** installed in WSL
- Your Windows “UK AGM CMD” folder containing each year’s subfolder (e.g.  
  `C:\Users\<WindowsUser>\Google Drive\UK AGM CMD\AGM YYYYMMDD`)  

---

## 2. If you already have this repo locally, skip the clone step

If you haven’t yet cloned the repository, do:
```bash
cd ~/whatever/path/you/keep/repos
git clone git@github.com:RS1987X/uk-performance-review.git
cd uk-performance-review

---

## 3. Create and activate a Python virtual enviroment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

--

## 4. Create a symlink to your Windows "UK AGM CMD" folder (one-time per machine)
ln -s "/mnt/c/Users/richa/Google Drive/UK AGM CMD" ~/uk-agm-cmd-folder

## 5. Create the new AGM folder and 
On Windows, create the new years agm folder:
ex. C:\Users\<WindowsUser>\Google Drive\UK AGM CMD\AGM 20260630

Put the following files in the folder 
    -Raw transactions file, add the column "Portfolio" manually
        to know which sub portlio did the transaction
    -File with starting positions
    -For symbols you know are missing in yahoo finance put
         the price histories in csv files in missing data from yahoo
         directory. Name them by the ISIN code 


## 6. Update src/config.py for the current year
#Keep BASE_DIR unchanged
BASE_DIR = Path.home() / "uk-agm-cmd-folder"
#Update CURRENT_AGM_FOLDER
CURRENT_AGM_FOLDER = "AGM 20250531"
AGM_DIR = BASE_DIR / CURRENT_AGM_FOLDER

TRANSACTIONS_CSV          = AGM_DIR / "transactions UK FKF.csv"
ISIN_MAPPING_CSV          = AGM_DIR / "isin_to_ticker_map.csv"
FORMATTED_TRANSACTIONS_CSV = AGM_DIR / "formatted_transactions.csv"
PRICES_CSV                = AGM_DIR / "closing_prices.csv"
START_POSITIONS_CSV       = AGM_DIR / "starting_position.csv"
MANUAL_CSV_FOLDER         = AGM_DIR / "missing data from yahoo"

OUTPUT_DIR = AGM_DIR / "charts"

## 7. Run the analysis
python run_all.py

this reads CSVs from
~/uk-agm-cmd-folder/AGM YYYYMMDD

and writes charts into
~/uk-agm-cmd-folder/AGM YYYYMMDD/charts

Charts and tables for the new year will appear in ~/uk-agm-cmd-folder/AGM YYYYMMDD/charts/.


## 8. Git best practices
Ensure .gitignore includes:
.venv/
__pycache__/
*.py[cod]
.vscode/
.idea/

After adding or updating packages:
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update dependencies"
git push
Commit changes to config.py when you switch the year or adjust folder names.```