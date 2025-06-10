
import sys
import traceback


# Make sure `src/` is on the import path
from pathlib import Path
workspace = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(workspace))

from src import build_mapping, format_transactions, fetch_prices, compute_pnls, metrics, plotting_main#, compute_performance_metrics
import src.config as config
import src.add_sub_portfolio_labels as add_sub_portfolio_labels


def main():
    print("▶ 1/4  Building ISIN→ticker mapping...")
    try:
        build_mapping.main()
    except Exception as e:
        print(f"⚠️  Error in build_mapping: {e}")
        traceback.print_exc()

    print("▶ 2/4  Formatting transactions...")
    print("▶ 3/5  Adding sub-portfolio labels…")
    # will launch Streamlit and exit if labels missing
    add_sub_portfolio_labels.run_streamlit_editor()
    # once labels exist, merge and get DataFrame for downstream steps
    add_sub_portfolio_labels.run_merge_if_ready()
    
    try:
        format_transactions.main()
    except Exception as e:
        print(f"⚠️  Error in format_transactions: {e}")
        traceback.print_exc()

    print("▶ 3/4  Fetching prices...")
    try:
        fetch_prices.main()
    except Exception as e:
        print(f"⚠️  Error in fetch_prices: {e}")
    traceback.print_exc()

    print("▶ 4/4  Computing PnL & generating charts...")
    # try:
    #     results_dict, uk_cum, uk_dd = compute_pnls.main()
    #     #compute_pnls.main()
    # except Exception as e:
    #     print(f"⚠️  Error in compute_pnls: {e}")
    #     traceback.print_exc()

    
    plotting_main.main()

    print("\n✅ All steps completed successfully.")

if __name__ == "__main__":
    main()