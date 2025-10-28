import pandas as pd

beh = pd.read_csv("MAR24_preview_merged.csv")
acc = pd.read_csv("Accident_preview.csv")

print("\nBEHAVIOUR COLUMNS:")
print(list(beh.columns))

print("\nACCIDENT COLUMNS:")
print(list(acc.columns))

print("\nFIRST 5 ROWS OF BEHAVIOUR:")
print(beh.head())

print("\nFIRST 5 ROWS OF ACCIDENT:")
print(acc.head())
