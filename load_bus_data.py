import pandas as pd
import numpy as np

# =========================
# 1. CONFIG: filenames
# =========================
# Make sure these match the actual filenames in your folder.
FILE_MAR24_PART1 = "./raw_datasets/Mar24_part1.xlsx"
FILE_MAR24_PART2 = "./raw_datasets/Mar24_part2.xlsx"
FILE_ACCIDENTS   = "./raw_datasets/Traffic Accident Database 2024.xlsx"

FILE_MAR24_MERGED = "./cleaned_datasets/Mar24_merged_cleaned.csv"
FILE_ACCIDENTS_MERGED = "./cleaned_datasets/Traffic Accident Database 2024 Cleaned.csv"

# =========================
# 2. LOAD METRICS DATA
# =========================
def load_driver_metrics_data():
    df_list = []
    max_no_rows = 1500

    mar24_excel_file1_sheets = pd.read_excel(FILE_MAR24_PART1, sheet_name=None)
    for sheet_name, sheet_df in mar24_excel_file1_sheets.items():
        sheet_df = sheet_df.head(max_no_rows)
        sheet_df.columns = (sheet_df.columns.str.strip().str.lower().str.replace(' ', '_', regex=True))
        sheet_df["driver"] = sheet_df["driver"].astype('string')
        sheet_df = sheet_df[sheet_df["driver"] != ""].copy()
        sheet_df["driver"] = sheet_df["driver"].astype(float).astype(int)
        df_list.append(sheet_df)
        print(f"Excel {sheet_name} has {len(sheet_df)} rows with valid driver ID.")

    mar24_excel_file2_sheets = pd.read_excel(FILE_MAR24_PART2, sheet_name=None)
    for sheet_name, sheet_df in mar24_excel_file2_sheets.items():
        sheet_df = sheet_df.head(max_no_rows)
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
# 4. LOAD ACCIDENTS DATA
# =========================
def load_accident_data():
    df_accidents_list = []

    accident_excel_file_sheets = pd.read_excel(FILE_ACCIDENTS, sheet_name=None)
    for sheet_name, sheet_df in accident_excel_file_sheets.items():
    #format column names
        sheet_df.columns = (sheet_df.columns.str.strip().str.lower().str.replace(' ', '_', regex=True))
        #ensure employee_no values all string, and they are all digits , then convert to int
        sheet_df['employee_no'] = sheet_df['employee_no'].astype('string')
        sheet_df = sheet_df[sheet_df["employee_no"] != ""].copy()
        sheet_df['employee_no'] = sheet_df['employee_no'].replace(r'\D+', '', regex=True)
        sheet_df['employee_no'] = sheet_df['employee_no'].astype(int)
        #ensure date of accident has no brackets
        sheet_df['date_of_accident'] = sheet_df['date_of_accident'].astype('string')
        sheet_df['date_of_accident'] = sheet_df['date_of_accident'].str.replace(r'\(.*?\)', '', regex=True)
        sheet_df['date_of_accident'] = pd.to_datetime(sheet_df['date_of_accident'], format='mixed')
        df_accidents_list.append(sheet_df)
        print(f"Excel {sheet_name} has {len(sheet_df)} rows with valid employee_no.")

    df_accidents = pd.concat(df_accidents_list, ignore_index=True)
    return df_accidents

# =========================
# 5. FILTER ACCIDENTS DATA BY SELECTED COLUMNS
# =========================
def filter_accident_columns(column_list, df):
    df = df[column_list].copy()
    df = df.sort_values(by="date_of_accident")
    return df

# =========================
# 6. QUICK INSPECTION
# =========================
def inspect_data(metrics_df, accidents_df):
    print("=== FIRST 10 ROWS: Metrics ===")
    print(metrics_df.head(10))

    print("\n=== LAST 10 ROWS: Metrics ===")
    print(metrics_df.tail(10))

    print("\n=== FIRST 10 ROWS: Accidents ===")
    print(accidents_df.head())

    print("\n=== LAST 10 ROWS: Accidents ===")
    print(accidents_df.tail())


# =========================
# MAIN
# =========================
def main():
    raw_metrics_df = load_driver_metrics_data()
    print(f"\nThe entire excel file has {len(raw_metrics_df)} alarm event instances with valid driver ID rows.\n")

    selected_metrics_columns = ["driver", "alarm_type", "alarm_duration"]
    filtered_metrics_df = filter_driver_metrics_columns(selected_metrics_columns, raw_metrics_df)

    raw_accidents_df = load_accident_data()
    print(f"\nTotal number of accidents from Jan-May 2024: {len(raw_accidents_df)}.\n")

    selected_accidents_cols = ["date_of_accident", "employee_no"]
    filtered_accidents_df = filter_accident_columns(selected_accidents_cols, raw_accidents_df)

    inspect_data(filtered_metrics_df, filtered_accidents_df)

    if len(filtered_metrics_df) > 0 and len(filtered_accidents_df) > 0:
        filtered_metrics_df.to_csv(FILE_MAR24_MERGED, index=False)
        print(f"\nSaved {FILE_MAR24_MERGED} .")

        filtered_accidents_df.to_csv(FILE_ACCIDENTS_MERGED, index=False)
        print(f"\nSaved {FILE_ACCIDENTS_MERGED} .")

        print("\nDone. You can open the CSVs in Excel to inspect.")


if __name__ == "__main__":
    main()
