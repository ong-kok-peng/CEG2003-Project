import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# Load the merged dataset
# -------------------------------------------------
df = pd.read_csv("Driver_Risk_Dataset.csv")

print("\n=== BASIC DATA OVERVIEW ===")
print(df.info())
print("\n=== SUMMARY STATISTICS ===")
print(df.describe())
print("\n=== At-Risk Distribution ===")
print(df["At_Risk"].value_counts())

# -------------------------------------------------
# 1️⃣ Histogram: Alarm Events per Driver
# -------------------------------------------------
plt.figure(figsize=(8,5))
plt.hist(df["num_alarm_rows"], bins=30, color="#1f77b4", alpha=0.7)
plt.title("Distribution of Alarm Events per Driver", fontsize=14)
plt.xlabel("Number of Alarm Events", fontsize=12)
plt.ylabel("Driver Count", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 2️⃣ Compare Average Behaviour of At-Risk vs Safe Drivers
# -------------------------------------------------
plt.figure(figsize=(8,5))
group_means = df.groupby("At_Risk")[["speed(km_h)", "alarm_duration", "num_alarm_rows"]].mean()
group_means.plot(kind="bar", figsize=(8,5), color=["#2ca02c", "#d62728"])
plt.title("Average Behaviour Metrics: At-Risk vs Safe Drivers", fontsize=14)
plt.xlabel("At_Risk (0=Safe, 1=Accident)", fontsize=12)
plt.ylabel("Average Value", fontsize=12)
plt.legend(["Speed (km/h)", "Alarm Duration", "Num Alarm Rows"])
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 3️⃣ Correlation Heatmap (to spot strong relationships)
# -------------------------------------------------
plt.figure(figsize=(6,4))
sns.heatmap(df[["speed(km_h)", "alarm_duration", "num_alarm_rows", "At_Risk"]].corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap", fontsize=13)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 4️⃣ Scatter Plot: Alarm Rows vs Alarm Duration (color by At_Risk)
# -------------------------------------------------
plt.figure(figsize=(7,5))
sns.scatterplot(
    data=df,
    x="num_alarm_rows",
    y="alarm_duration",
    hue="At_Risk",
    palette={0:"skyblue", 1:"red"},
    alpha=0.6
)
plt.title("Drivers with More Alarms Tend to Have Longer Alarm Durations", fontsize=13)
plt.xlabel("Number of Alarm Events", fontsize=12)
plt.ylabel("Total Alarm Duration", fontsize=12)
plt.legend(title="At Risk (1=Yes, 0=No)")
plt.tight_layout()
plt.show()
