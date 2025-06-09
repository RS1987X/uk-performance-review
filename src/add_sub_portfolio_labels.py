# add_sub_portfolio_labels.py
import streamlit as st
import pandas as pd
import os
import src.config as config
import subprocess
import sys
from pathlib import Path
from src.utils import read_messy_tab_file 
# RAW_PATH   = "original_data.csv"
# ANNOT_PATH = "annotations.csv"
# OUTPUT_PATH = "augmented_data.csv"

@st.cache_data
def load_raw():
    return read_messy_tab_file(config.RAW_TX_PATH)

@st.cache_data
def load_annotations():
    # 1) If the raw transactions already include subportfolio assignments, grab them
    raw_df = read_messy_tab_file(config.RAW_TX_PATH)
    if "subportfolio" in raw_df.columns:
        # only keep id + existing labels
        return raw_df[["Id", "subportfolio"]].drop_duplicates().reset_index(drop=True)

    # 2) Else if we’ve previously saved a labels file, load that
    if os.path.exists(config.SUBPORTF_PATH):
        ann = pd.read_csv(config.SUBPORTF_PATH, dtype=str)
        # ensure every ID shows up, even if unlabeled
        template = raw_df[["Id"]].drop_duplicates().sort_values("Id")
        return template.merge(ann, on="Id", how="left")

    # 3) Otherwise first run: create blank template
    template = raw_df[["Id"]].drop_duplicates().sort_values("Id")
    template["subportfolio"] = ""
    return template

def launch_annotation_ui():
    print("🔍 No subportfolio file found. Launching Streamlit editor…\n"
          "   (When you’ve filled and saved, Ctrl+C to stop Streamlit and re-run this script.)")
    try:
        # This will block until you hit Ctrl+C
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", __file__],
            check=False
        )
    except KeyboardInterrupt:
        # Streamlit server was interrupted by Ctrl+C
        print("✍️  Detected Ctrl+C — resuming pipeline…")
    # now control returns here, and the script can continue


def run_streamlit_editor():
    #if not Path(config.SUBPORTF_PATH).exists():
    launch_annotation_ui()


def run_merge_if_ready():
    if Path(config.SUBPORTF_PATH).exists():
        return merge_annotations()
    return None

def merge_annotations():
    import pandas as pd
    print("📥  Loading raw transactions…")
    df = read_messy_tab_file(config.RAW_TX_PATH)

    print("📥  Loading Sub-portfolio file…")
    ann = pd.read_csv(config.SUBPORTF_PATH, dtype=str)
    print("🔗  Merging on ‘Id’…")
    merged = df.merge(ann, on="Id", how="left")
    merged.to_csv(config.AUGMENTED_TX_PATH, index=False)
    print(f"✅  Augmented transactions → {config.AUGMENTED_TX_PATH}")
    return merged

def main():
    st.set_page_config(layout="wide")
    st.title("Add Sub-portfolio labels")
    raw = load_raw()
    ann = load_annotations()
    
    # Merge annotations into raw
    display_cols = ["Id", "Affärsdag", "Transaktionstyp", "Värdepapper", "ISIN", "Antal", "Kurs"]
    df = raw[display_cols].drop_duplicates().merge(ann, on="Id", how="left")
    
    CANDIDATE_SUBPORTFOLIOS = ["BURMAN","BERGMAN", "GUNNARSSON", "JAEGERSTAD", "LINDHE",
                               "SAFFAR", "SJOGREN", "SODERLUND", "STEFFEN",
                               "SKUGG"]
    
    #df = raw[["Id"]].drop_duplicates().merge(ann, on="Id", how="left")
    #edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "subportfolio": st.column_config.SelectboxColumn(
                "Subportfolio",
                help="Assign to one of the defined sub-portfolios",
                options=CANDIDATE_SUBPORTFOLIOS,
                required=False,
            )
        }
)

    if st.button("Save annotations"):
        # persist only the ID + subjective column
        output = edited[["Id", "subportfolio"]]
        output.to_csv(config.SUBPORTF_PATH, index=False)
        st.success(f"Saved {len(output)} annotations to {config.SUBPORTF_PATH}")
        # also write the final merged dataset
        merged = raw.merge(output, on="Id", how="left")
        merged.to_csv(config.AUGMENTED_TX_PATH, index=False)
        st.info(f"Augmented dataset → {config.AUGMENTED_TX_PATH}")

if __name__ == "__main__":
    main()
