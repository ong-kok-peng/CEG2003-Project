import pandas as pd

acc = pd.read_csv("Accident_preview.csv")
beh = pd.read_csv("MAR24_preview_merged.csv")

print("ACCIDENT COLUMNS:", list(acc.columns))
print("\nUNIQUE SAMPLE Employee No (first 20 non-null):")
print(acc["Employee No"].dropna().head(20))

print("\nUNIQUE SAMPLE Service No (first 20 non-null):")
print(acc["Service No"].dropna().head(20))

print("\nBEHAVIOUR `Driver` sample:")
print(beh["Driver"].dropna().head(20))
