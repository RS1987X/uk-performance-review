# add_sub_portfolio_labels.py

import streamlit as st
import pandas as pd
import os
import src.config as config
import subprocess
import sys
from pathlib import Path
from src.utils import read_messy_tab_file
import time
import platform
import webbrowser

ALLOC_FILE = Path(config.SUBPORTF_PATH)



@st.cache_data
def load_raw():
    return read_messy_tab_file(config.RAW_TX_PATH)


def load_allocations():
    """
    Returns a DataFrame with columns: ['Id', 'Sub-Portfolio', 'Percentage'].

    Handles two on-disk formats:
      - Legacy: ['Id', 'subportfolio'] → each gets 100%
      - New   : ['Id', 'Sub-Portfolio', 'Percentage']
    """
    raw_df = read_messy_tab_file(config.RAW_TX_PATH)
    cols = set(raw_df.columns)

    # # New format
    # if {"Id", "Sub-Portfolio", "Percentage"}.issubset(cols):
    #     df = raw_df.copy()
    #     df["Id"] = df["Id"].astype(str)
    #     df["Sub-Portfolio"] = df["Sub-Portfolio"].astype(str)
    #     df["Percentage"] = df["Percentage"].astype(float)
    #     return df[["Id", "Sub-Portfolio", "Percentage"]]

    # # Legacy format with in-file labels
    # if "Sub-Portfolio" in cols:
    #     df = raw_df[["Id", "Sub-Portfolio", "Antal"]].drop_duplicates().copy()
    #     df["Sub-Portfolio"] = df["Sub-Portfolio"].astype(str)
    #     #df["Percentage"] = 100.0
    #     return df[["Id", "Sub-Portfolio"]]

    # Try to load saved annotations if available
    if os.path.exists(config.SUBPORTF_PATH):
        ann = pd.read_csv(config.SUBPORTF_PATH)
        # Ensure 'Id' is of the same type in both DataFrames
        
        raw_df["Id"] = raw_df["Id"].astype(str)
        ann["Id"] = ann["Id"].astype(str)  # Convert 'Id' in ann to string

        
        template = raw_df[["Id"]].drop_duplicates().sort_values("Id")
        merged = template.merge(ann, on="Id", how="left")
        # Ensure 'Sub-Portfolio' is a string
        merged["Sub-Portfolio"] = merged["Sub-Portfolio"].fillna("").astype(str)
        #merged["Percentage"] = 100.0
        return merged[["Id", "Sub-Portfolio", "Antal"]]
    else:
        return None
    # # First run: blank template
    # template = raw_df[["Id"]].drop_duplicates().sort_values("Id")
    # template["Sub-Portfolio"] = ""
    # #template["Percentage"] = 0
    # return template[["Id", "Sub-Portfolio"]]

    # If none of the above match, raise error (shouldn’t reach here now)
    raise ValueError(
        f"Unexpected columns in input: {raw_df.columns.tolist()}"
    )

def save_allocations(df: pd.DataFrame):
    """
    Persist the full allocations table (new format) back to CSV.
    """
    # Ensure correct column order
    out = df[["Id","Sub-Portfolio", "Antal"]].copy()
    out.to_csv(ALLOC_FILE, index=False)


def launch_annotation_ui():
    """
    Fallback launcher if someone explicitly wants to spin up a separate process.
    """
    st.warning("No sub-portfolio allocations file found; launching Streamlit…")
    cmd = [sys.executable, "-m", "streamlit", "run", __file__]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    url = f"http://localhost:{config.STREAMLIT_PORT or 8501}"
    if sys.platform.startswith("linux") and "microsoft" in platform.uname().release.lower():
        subprocess.run(["explorer.exe", url], check=False)
    else:
        webbrowser.open(url, new=2)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()

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
    st.title("Shared Sub-Portfolio Allocations")

    raw = load_raw()
    raw["Id"] = raw["Id"].astype(str)

    all_allocs = load_allocations()

    # If no allocations are available, create an empty DataFrame with the required structure
    if all_allocs is None:
        all_allocs = pd.DataFrame(columns=["Id", "Sub-Portfolio", "Antal"])


    # Define transaction fields to display
    tx_fields = ["Id", "Affärsdag", "Transaktionstyp", "Värdepapper", "ISIN", "Antal", "Kurs"]

    # Merge raw with allocations
    merged = raw[tx_fields].drop_duplicates().merge(
        all_allocs, on="Id", how="left", suffixes=("_raw", "_alloc")
    )

    # Resolve the 'Antal' column
    merged["Antal"] = merged["Antal_alloc"].fillna(merged["Antal_raw"]).fillna(0)

    # Drop unnecessary columns
    merged = merged.drop(columns=["Antal_raw", "Antal_alloc"])

    # Fill defaults for other columns
    merged["Sub-Portfolio"] = merged["Sub-Portfolio"].fillna("")

    # Add a "Select Row" column
    merged["Select Row"] = False

    st.markdown("You can split each transaction into multiple sub-portfolios by adding multiple rows with the same `Id`.")

    if "edited_df" not in st.session_state:
        st.session_state["edited_df"] = merged.copy()

    # Display the editable table
    edited_df = st.data_editor(
    st.session_state["edited_df"],  # Use the updated session state DataFrame
    num_rows="dynamic",
    use_container_width=True,
    disabled=["Affärsdag", "Transaktionstyp", "Värdepapper", "ISIN", "Kurs"],
    column_config={
        "Sub-Portfolio": st.column_config.SelectboxColumn(
            "Sub-Portfolio",
            options=config.SUB_PORTFOLIO_NAMES
        ),
        "Antal": st.column_config.NumberColumn(
            "Antal",
            min_value=0,
            step=0.1,
            format="%.1f"
        ),
        "Select Row": st.column_config.CheckboxColumn(
            "Select Row",
            help="Select a row to add a shared transaction."
        )
    }
)

    # Check if a row is selected
    selected_rows = edited_df[edited_df["Select Row"]]
    if not selected_rows.empty:
        selected_index = selected_rows.index[0]  # Get the first selected row index
        selected_row = edited_df.iloc[selected_index]

        st.markdown(f"### Add Shared Transaction for ID: {selected_row['Id']}")
        with st.form("shared_transaction_form"):
            # Allow users to specify multiple sub-portfolios
            sub_portfolios = st.multiselect(
                "Sub-Portfolios",
                options=config.SUB_PORTFOLIO_NAMES,
                help="Select one or more sub-portfolios for the shared transactions."
            )
            quantities = st.text_input(
                "Number of shares (comma-separated)",
                help="Enter number of shares for each sub-portfolio, separated by commas. The total must match the original quantity."
            )
            submitted = st.form_submit_button("Add Shared Transactions")

            if submitted:
                # Parse the quantities
                try:
                    quantities = [float(q.strip()) for q in quantities.split(",")]
                except ValueError:
                    st.error("⚠️ Invalid quantities. Please enter valid numbers separated by commas.")
                    return

                # Validate the input
                if len(sub_portfolios) != len(quantities):
                    st.error("⚠️ The number of sub-portfolios and quantities must match.")
                    return
                if abs(sum(quantities) - float(selected_row["Antal"])) > 1e-6:
                    st.error(f"⚠️ The total quantities must match the original quantity ({selected_row['Antal']}).")
                    return

                # Create new rows for each sub-portfolio
                new_rows = []
                for sub_portfolio, quantity in zip(sub_portfolios, quantities):
                    new_row = selected_row.copy()
                    new_row["Sub-Portfolio"] = sub_portfolio
                    new_row["Antal"] = quantity
                    new_row["Select Row"] = False  # Reset selection
                    new_rows.append(new_row)

                # Update the session state table
                st.session_state["edited_df"] = pd.concat(
                    [
                        st.session_state["edited_df"].iloc[:selected_index + 1],
                        pd.DataFrame(new_rows),
                        st.session_state["edited_df"].iloc[selected_index + 1:]
                    ],
                    ignore_index=True
                )

                # Reset the "Select Row" column
                st.session_state["edited_df"]["Select Row"] = False
                #Remove the original row
                st.session_state["edited_df"] = st.session_state["edited_df"].drop(index=selected_index).reset_index(drop=True)

                st.success("Shared transaction added successfully!")

    # Validate all Ids
    validation = (
        edited_df.dropna(subset=["Sub-Portfolio"])
        .groupby("Id")["Antal"]
        .sum()
        .reset_index()
    )
    invalid = validation[abs(validation["Antal"] - raw.set_index("Id")["Antal"]).abs() > 1e-6]

    if not invalid.empty:
        st.error("⚠️ Some transactions don't sum to 100%:")
        st.dataframe(invalid)
    else:
        st.success("✅ All transaction allocations sum to 100%.")

    if st.button("Save All Allocations"):
        to_save = edited_df.dropna(subset=["Sub-Portfolio"]).copy()
        to_save = to_save[["Id", "Sub-Portfolio", "Antal"]]
        to_save["Antal"] = pd.to_numeric(to_save["Antal"], errors="coerce").fillna(0)

        
        to_save["Id"] = to_save["Id"].astype(str)
        to_save["Sub-Portfolio"] = to_save["Sub-Portfolio"].astype(str)
        to_save["Antal"] = to_save["Antal"].astype(float)
        
        #to_save["Percentage"] = to_save["Percentage"].astype(float)
        save_allocations(to_save)
        st.success("Allocations saved.")

    if st.checkbox("Show raw transaction table"):
        st.dataframe(raw, use_container_width=True)

def run_streamlit_editor():
    """
    Entry point for external scripts (e.g. run_all.py).
    """
    #main()
    launch_annotation_ui()

if __name__ == "__main__":
    #launch_annotation_ui()
    main()
