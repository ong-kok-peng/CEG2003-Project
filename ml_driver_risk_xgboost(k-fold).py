import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
df = pd.read_csv("Driver_Risk_Dataset.csv")

# Choose feature columns and target
feature_cols = ["speed(km_h)", "alarm_duration", "num_alarm_rows"]
X = df[feature_cols].values
y = df["At_Risk"].values

print("Class distribution (original):")
print(df["At_Risk"].value_counts(), "\n")

# -------------------------------------------------
# Stratified K-Fold Cross-Validation with XGBoost
# -------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_y_true = []
all_y_pred = []

fold = 1
for train_idx, test_idx in skf.split(X, y):
    print(f"\n=== Fold {fold} (XGBoost + SMOTE) ===")
    fold += 1

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("Class distribution BEFORE SMOTE (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))

    # ------- Pipeline: SMOTE -> XGBoost -------
    smote = SMOTE(random_state=42)

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1
        # NOTE: we usually do NOT use scale_pos_weight together with SMOTE
    )

    model = Pipeline(steps=[
        ('smote', smote),
        ('xgb', xgb)
    ])

    # Fit on TRAIN only (SMOTE happens inside fit)
    model.fit(X_train, y_train)

    # Just predict on raw test fold (no SMOTE on test)
    y_pred = model.predict(X_test)

    print("Class distribution AFTER SMOTE (train) - approximate:")
    X_res, y_res = smote.fit_resample(X_train, y_train)
    unique_res, counts_res = np.unique(y_res, return_counts=True)
    print(dict(zip(unique_res, counts_res)))

    print("\nClassification report (this fold):")
    print(classification_report(y_test, y_pred, digits=3))

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# -------- Overall performance across all folds --------
print("\n=== OVERALL RESULTS (XGBoost + SMOTE, all folds combined) ===")
print(classification_report(all_y_true, all_y_pred, digits=3))

print("Confusion matrix (all folds combined):")
print(confusion_matrix(all_y_true, all_y_pred))
