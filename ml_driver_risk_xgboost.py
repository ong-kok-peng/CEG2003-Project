import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
df = pd.read_csv("Driver_Risk_Dataset.csv")

X = df[["speed(km_h)", "alarm_duration", "num_alarm_rows"]]
y = df["At_Risk"]

print("Class distribution (original):")
print(y.value_counts(), "\n")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -------------------------------------------------
# Train XGBoost model with built-in class weighting
# -------------------------------------------------
# Calculate scale_pos_weight = (# of safe) / (# of risky)
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=scale_pos_weight,  # balance classes
    eval_metric='logloss',
    use_label_encoder=False
)

model.fit(X_train, y_train)

# -------------------------------------------------
# Evaluate model
# -------------------------------------------------
y_pred = model.predict(X_test)

print("=== CLASSIFICATION REPORT (XGBoost) ===")
print(classification_report(y_test, y_pred, digits=3))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
plt.title("Confusion Matrix (XGBoost)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Feature Importance
# -------------------------------------------------
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n=== FEATURE IMPORTANCE (XGBoost) ===")
print(importance)

plt.figure(figsize=(5,3))
sns.barplot(x='Importance', y='Feature', data=importance, palette="plasma")
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()
