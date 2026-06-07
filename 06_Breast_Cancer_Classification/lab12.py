# ============================================================
# Lab 12 – Random Forest Classifier vs Logistic Regression
# Dataset: Breast Cancer Wisconsin (Diagnostic) – UCI
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame  # includes features + target
print("Dataset shape:", df.shape)
print("Columns:\n", df.columns)

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# ------------------------------------------------------------
# 2. TRAIN–TEST SPLIT
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ------------------------------------------------------------
# 3. FEATURE SCALING (for Logistic Regression)
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 4. MODEL DEFINITIONS
# ------------------------------------------------------------

# Random Forest Classifier
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"  # handle mild imbalance
)

# Logistic Regression (comparison model)
log_clf = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)

# ------------------------------------------------------------
# 5. TRAIN MODELS
# ------------------------------------------------------------
rf_clf.fit(X_train, y_train)
log_clf.fit(X_train_scaled, y_train)

# ------------------------------------------------------------
# 6. PREDICTIONS & PROBABILITIES
# ------------------------------------------------------------
# Random Forest
y_pred_rf = rf_clf.predict(X_test)
y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]

# Logistic Regression
y_pred_log = log_clf.predict(X_test_scaled)
y_prob_log = log_clf.predict_proba(X_test_scaled)[:, 1]

# ------------------------------------------------------------
# 7. METRIC FUNCTION
# ------------------------------------------------------------
def evaluate_model(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:\n",
          classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    return acc, prec, rec, f1, auc

# ------------------------------------------------------------
# 8. EVALUATE BOTH MODELS
# ------------------------------------------------------------
rf_metrics = evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)
log_metrics = evaluate_model("Logistic Regression", y_test, y_pred_log, y_prob_log)

# ------------------------------------------------------------
# 9. ROC CURVE COMPARISON
# ------------------------------------------------------------
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)

plt.figure(figsize=(7, 5))
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot(fpr_log, tpr_log, label="Logistic Regression")
plt.plot([0, 1], [0, 1], "k--", label="Baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Random Forest vs Logistic Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 10. FEATURE IMPORTANCE (Random Forest)
# ------------------------------------------------------------
importances = rf_clf.feature_importances_
feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTop 10 important features (Random Forest):")
print(feature_importance_df.head(10))
