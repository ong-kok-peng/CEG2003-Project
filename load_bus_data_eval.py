import pandas as pd
import numpy as np

# =========================
# 1. CONFIG: filenames
# =========================
# Make sure these match the actual filenames in your folder.
FILE_MAR24_PART1 = "./raw_datasets/Mar24_part1.xlsx"
FILE_MAR24_PART2 = "./raw_datasets/Mar24_part2.xlsx"

FILE_MAR24_MERGED_EVAL = "./cleaned_datasets/Mar24_merged_cleaned_eval.csv"

# =========================
# 2. LOAD METRICS DATA
# =========================
def load_driver_metrics_data(max_no_rows_per_sheet):
    df_list = []

    mar24_excel_file1_sheets = pd.read_excel(FILE_MAR24_PART1, sheet_name=None)
    for sheet_name, sheet_df in mar24_excel_file1_sheets.items():
        sheet_df = sheet_df.sample(n=max_no_rows_per_sheet)
        sheet_df.columns = (sheet_df.columns.str.strip().str.lower().str.replace(' ', '_', regex=True))
        sheet_df["driver"] = sheet_df["driver"].astype('string')
        sheet_df = sheet_df[sheet_df["driver"] != ""].copy()
        sheet_df["driver"] = sheet_df["driver"].astype(float).astype(int)
        df_list.append(sheet_df)
        print(f"Excel {sheet_name} has {len(sheet_df)} rows with valid driver ID.")

    mar24_excel_file2_sheets = pd.read_excel(FILE_MAR24_PART2, sheet_name=None)
    for sheet_name, sheet_df in mar24_excel_file2_sheets.items():
        sheet_df = sheet_df.sample(n=max_no_rows_per_sheet)
        sheet_df.columns = (sheet_df.columns.str.strip().str.lower().str.replace(' ', '_', regex=True))
        sheet_df["driver"] = sheet_df["driver"].astype('string')
        sheet_df = sheet_df[sheet_df["driver"] != ""].copy()
        sheet_df["driver"] = sheet_df["driver"].astype(float).astype(int)
        df_list.append(sheet_df)
        print(f"Excel {sheet_name} has {len(sheet_df)} rows with valid driver ID.")

    df_metrics = pd.concat(df_list, ignore_index=True)
    return df_metrics

# =========================
# 3. FILTER METRICS DATA BY SELECTED COLUMNS
# =========================
def filter_driver_metrics_columns(columns_list, df):
    df = df[columns_list].copy()
    df = df.sort_values(by="driver")
    return df

# =========================
# 4. QUICK INSPECTION
# =========================
def inspect_data(metrics_df):
    print("=== FIRST 10 ROWS: Metrics ===")
    print(metrics_df.head(10))

    print("\n=== LAST 10 ROWS: Metrics ===")
    print(metrics_df.tail(10))


# =========================
# MAIN
# =========================
def main():
    #randomly sample 100000 rows from the entire file_mar24 excel
    max_no_rows_whole = 100000
    no_of_sheets = 6
    max_no_rows_per_sheet = np.rint(max_no_rows_whole / no_of_sheets).astype(np.uint64)

    raw_metrics_df = load_driver_metrics_data(max_no_rows_per_sheet)
    print(f"\nThe entire excel file has {len(raw_metrics_df)} alarm event instances with valid driver ID rows.\n")

    selected_metrics_columns = ["driver", "alarm_type", "alarm_duration"]
    filtered_metrics_df = filter_driver_metrics_columns(selected_metrics_columns, raw_metrics_df)

    inspect_data(filtered_metrics_df)

    if len(filtered_metrics_df) > 0:
        filtered_metrics_df.to_csv(FILE_MAR24_MERGED_EVAL, encoding='utf-8', index=False)
        print(f"\nSaved {FILE_MAR24_MERGED_EVAL} .")
        print("\nDone. You can open the CSV in Excel to inspect.")


if __name__ == "__main__":
    main()
