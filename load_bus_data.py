import pandas as pd

# =========================
# 1. CONFIG: filenames
# =========================
# Make sure these match the actual filenames in your folder.
FILE_MAR24_PART1 = "Mar24_part1.xlsx"
FILE_MAR24_PART2 = "Mar24_part2.xlsx"
FILE_ACCIDENTS   = "Traffic Accident Database 2024.xlsx"

# Optional: limit rows for the big MAR24 files so VS Code won't die from memory usage
PREVIEW_ROWS = 1000   # you can increase later


# =========================
# 2. LOAD DATA
# =========================
def load_data():
    print("Loading MAR24 Part 1...")
    mar1 = pd.read_excel(FILE_MAR24_PART1, nrows=PREVIEW_ROWS)

    print("Loading MAR24 Part 2...")
    mar2 = pd.read_excel(FILE_MAR24_PART2, nrows=PREVIEW_ROWS)

    print("Loading Accident Database (full)...")
    accidents = pd.read_excel(FILE_ACCIDENTS)

    return mar1, mar2, accidents


# =========================
# 3. QUICK INSPECTION
# =========================
def inspect_data(mar1, mar2, accidents):
    print("\n=== SHAPE CHECK ===")
    print("Mar24 Part 1 shape:", mar1.shape)
    print("Mar24 Part 2 shape:", mar2.shape)
    print("Accident DB shape:", accidents.shape)

    print("\n=== COLUMN NAMES ===")
    print("Mar24 Part 1 columns:", list(mar1.columns))
    print("Mar24 Part 2 columns:", list(mar2.columns))
    print("Accident DB columns:", list(accidents.columns))

    print("\n=== FIRST 5 ROWS: MAR24 Part 1 ===")
    print(mar1.head())

    print("\n=== FIRST 5 ROWS: Accident DB ===")
    print(accidents.head())


# =========================
# 4. MERGE PART1 + PART2
# =========================
def combine_mar_data(mar1, mar2):
    # Check if both MAR24 parts have the same columns
    if list(mar1.columns) == list(mar2.columns):
        mar_all = pd.concat([mar1, mar2], ignore_index=True)
        print("\nMerged MAR24 preview shape:", mar_all.shape)
    else:
        print("\nWARNING: Column names don't match between Part1 and Part2.")
        print("Part1:", list(mar1.columns))
        print("Part2:", list(mar2.columns))
        mar_all = None

    return mar_all


# =========================
# 5. MAIN
# =========================
def main():
    mar1, mar2, accidents = load_data()
    inspect_data(mar1, mar2, accidents)

    mar_all = combine_mar_data(mar1, mar2)

    # OPTIONAL: save preview outputs for the rest of the team to look at
    if mar_all is not None:
        mar_all.to_csv("MAR24_preview_merged.csv", index=False)
        print('\nSaved merged MAR24 preview -> MAR24_preview_merged.csv')

    accidents.to_csv("Accident_preview.csv", index=False)
    print('Saved accident data preview -> Accident_preview.csv')

    print("\nDone. You can open the CSVs in Excel to inspect.")


if __name__ == "__main__":
    main()
