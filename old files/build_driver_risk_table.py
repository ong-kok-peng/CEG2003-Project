import pandas as pd

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
FILE_BEHAVIOR = "MAR24_preview_merged.csv"          # alarm / behaviour data
FILE_ACCIDENT = "Accident_preview.csv"              # accident records
OUTPUT_MERGED = "Driver_Risk_Dataset.csv"           # final output

BEHAVIOR_DRIVER_COL_RAW = "Driver"        # from MAR24_preview_merged.csv
ACCIDENT_DRIVER_COL_RAW = "Employee No"   # from Accident_preview.csv


def norm(name: str) -> str:
    """Normalize column names to snake-ish format so merging is easier."""
    return (
        name.strip()
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace("\n", "_")
    )


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [norm(c) for c in df.columns]
    return df


def load_data():
    beh_raw = pd.read_csv(FILE_BEHAVIOR)
    acc_raw = pd.read_csv(FILE_ACCIDENT)

    beh = clean_cols(beh_raw)
    acc = clean_cols(acc_raw)

    print("[INFO] Behaviour columns (cleaned):", list(beh.columns))
    print("[INFO] Accident columns  (cleaned):", list(acc.columns))

    return beh, acc


def parse_datetimes(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass
    return df


def build_behaviour_features(beh: pd.DataFrame) -> pd.DataFrame:
    """
    We aggregate alarm behaviour per driver:
    - sum numeric risk features (speed, alarm duration)
    - count how many alarm rows they triggered
    """

    driver_col_clean = norm(BEHAVIOR_DRIVER_COL_RAW)  # "Driver" -> "driver"
    if driver_col_clean not in beh.columns:
        raise KeyError(
            f"[ERROR] '{driver_col_clean}' not found in behaviour data columns {list(beh.columns)}"
        )

    # Identify which numeric columns should be aggregated.
    # We'll ignore obvious non-feature columns.
    skip_keywords = [
        "driver", "time", "date", "line", "route",
        "monthyear", "weekyear", "startdate", "alarm_type"
    ]

    feature_cols = []
    for col in beh.columns:
        if col == driver_col_clean:
            continue
        if any(k in col for k in skip_keywords):
            continue
        if pd.api.types.is_numeric_dtype(beh[col]):
            feature_cols.append(col)

    # Aggregate sums of numeric features by driver
    if feature_cols:
        agg_numeric = (
            beh.groupby(driver_col_clean)[feature_cols]
               .sum()
               .reset_index()
        )
    else:
        # fallback: empty frame with just driver col
        agg_numeric = (
            beh[[driver_col_clean]]
            .drop_duplicates()
            .copy()
        )

    # Add count of alarm rows (how many events were recorded for this driver)
    row_count = (
        beh.groupby(driver_col_clean)
           .size()
           .reset_index(name="num_alarm_rows")
    )

    beh_agg = agg_numeric.merge(row_count, on=driver_col_clean, how="left")

    # Standardize driver_id: convert 16890.0 -> "16890"
    beh_agg["driver_id"] = (
        beh_agg[driver_col_clean]
        .astype(float)      # ensure numeric
        .astype("Int64")    # drop .0
        .astype(str)
    )

    # Drop the original driver column to avoid confusion
    beh_agg = beh_agg.drop(columns=[driver_col_clean])

    print("\n[DEBUG] Behaviour aggregate preview:")
    print(beh_agg.head())

    return beh_agg


def build_accident_flags(acc: pd.DataFrame) -> pd.DataFrame:
    """
    We build a table of drivers who have accidents:
    driver_id | At_Risk (1)
    """

    accident_driver_col_clean = norm(ACCIDENT_DRIVER_COL_RAW)  # "Employee No" -> "employee_no"
    if accident_driver_col_clean not in acc.columns:
        raise KeyError(
            f"[ERROR] '{accident_driver_col_clean}' not found in accident data columns {list(acc.columns)}"
        )

    # Keep unique Employee No values (this identifies the bus captain)
    flag = (
        acc[[accident_driver_col_clean]]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    # Convert to same string form as behaviour driver_id
    # (acc is already object like "19279", "20009", etc, but we force str anyway)
    flag["driver_id"] = flag[accident_driver_col_clean].astype(str)

    flag["At_Risk"] = 1
    flag = flag[["driver_id", "At_Risk"]]

    print("\n[DEBUG] Accident flag preview:")
    print(flag.head())

    return flag


def main():
    # 1. Load and clean column names
    beh, acc = load_data()

    # 2. Try to parse datetimes in behaviour (not critical but nice)
    beh = parse_datetimes(
        beh,
        cols=["alarm_start_time", "alarm_end_time", "startdate"]
    )

    # 3. Build per-driver behaviour summary
    beh_agg = build_behaviour_features(beh)

    # 4. Build accident-driven risk labels
    accident_flag = build_accident_flags(acc)

    # 5. Merge behaviour + risk
    final_df = beh_agg.merge(accident_flag, on="driver_id", how="left")

    # Drivers not in accident list get 0
    final_df["At_Risk"] = final_df["At_Risk"].fillna(0).astype(int)

    print("\n[DEBUG] Final merged dataset preview:")
    print(final_df.head())

    # 6. Save final dataset for ML / dashboard
    final_df.to_csv(OUTPUT_MERGED, index=False)
    print(f"\n[SAVED] {OUTPUT_MERGED}")
    print("Done.")


if __name__ == "__main__":
    main()
