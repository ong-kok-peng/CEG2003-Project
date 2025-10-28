import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
df = pd.read_csv("Driver_Risk_Dataset.csv")

X = df[["speed(km_h)", "alarm_duration", "num_alarm_rows"]]
y = df["At_Risk"]

print("Class distribution (original):")
print(y.value_counts(), "\n")

# Split BEFORE oversampling
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# Oversample minority class on TRAINING ONLY
# -------------------------------------------------
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

print("Class distribution AFTER SMOTE (train set only):")
print(y_train_bal.value_counts(), "\n")

# -------------------------------------------------
# Train model (still keep class_weight='balanced' to be extra safe)
# -------------------------------------------------
model = RandomForestClassifier(
    random_state=42,
    n_estimators=200,
    class_weight='balanced'
)
model.fit(X_train_bal, y_train_bal)

# Predict on untouched test set
y_pred = model.predict(X_test)

# -------------------------------------------------
# Evaluation
# -------------------------------------------------
print("=== CLASSIFICATION REPORT (SMOTE + weighted RF) ===")
print(classification_report(y_test, y_pred, digits=3))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
plt.title("Confusion Matrix (SMOTE + Weighted RF)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Feature Importance
# -------------------------------------------------
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n=== FEATURE IMPORTANCE (SMOTE + weighted RF) ===")
print(importances)

plt.figure(figsize=(5,3))
sns.barplot(x='Importance', y='Feature', data=importances)
plt.title("Feature Importance (SMOTE + Weighted RF)")
plt.tight_layout()
plt.show()
