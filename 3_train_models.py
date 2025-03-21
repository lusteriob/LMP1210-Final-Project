import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# === Load and merge features ===
boss_df = pd.read_pickle("features/boss_features_pads_style.pkl")
manual_df = pd.read_pickle("features/manual_features_pads_style.pkl")

# Remove PatientID columns if present
if "PatientID" in boss_df.columns:
    boss_df = boss_df.drop(columns=["PatientID"])
if "PatientID" in manual_df.columns:
    manual_df = manual_df.drop(columns=["PatientID"])

# Load labels (same for both)
y = pd.read_csv("features/boss_labels_pads_style.csv")["Label"]

# Ensure same patient order
assert boss_df.shape[0] == manual_df.shape[0], "Mismatched patient counts"

# === Stack features together ===
X_combined = np.hstack([
    boss_df.astype(np.float32).values,
    manual_df.astype(np.float32).values
])
n_boss = boss_df.shape[1]
n_manual = manual_df.shape[1]

# === Custom Transformer to switch feature subsets ===
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, mode="boss"):
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.mode == "boss":
            return X[:, :n_boss]
        elif self.mode == "manual":
            return X[:, n_boss:]
        elif self.mode == "combined":
            return X
        else:
            raise ValueError("Invalid mode for FeatureSelector: boss/manual/combined")

# === CV setup ===
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === Pipeline ===
pipeline = Pipeline([
    ("feature_selector", FeatureSelector()),
    ("scaler", StandardScaler()),
    ("svm", SVC(class_weight="balanced", probability=True))
])

# === Grid with feature selection + SVM params ===
param_grid = {
    "feature_selector__mode": ["boss", "manual", "combined"],
    "svm__C": [0.1, 1, 10],
    "svm__gamma": ["scale", 0.01],
    "svm__kernel": ["rbf"]
}

# === Output directory ===
model_dir = "models/svm_nested_cv_combined"
os.makedirs(model_dir, exist_ok=True)

# === Store metrics ===
results = []

print("🔁 Running 5x5 nested cross-validation (BOSS vs Manual vs Combined)...\n")

for fold_idx, (train_idx, test_idx) in enumerate(tqdm(outer_cv.split(X_combined, y), total=5, desc="Outer Folds")):
    print(f"\n🔄 Starting Outer Fold {fold_idx + 1}/5")

    X_train, X_test = X_combined[train_idx], X_combined[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=make_scorer(balanced_accuracy_score),
        cv=inner_cv,
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Save best model
    model_path = os.path.join(model_dir, f"svm_fold{fold_idx + 1}.joblib")
    joblib.dump(best_model, model_path)

    # Save metrics
    metrics = {
        "fold": fold_idx + 1,
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "best_params": grid.best_params_
    }
    results.append(metrics)

    print(f"\n✅ Fold {metrics['fold']} complete:")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  F1 Score:          {metrics['f1']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  ROC AUC:           {metrics['roc_auc']:.4f}")
    print(f"  Best Params:       {metrics['best_params']}\n")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(model_dir, "metrics_summary.csv"), index=False)

# Summary
print("✅ Nested CV Completed.\n")
print("📊 Average metrics across folds:")
print(results_df[["balanced_accuracy", "f1", "precision", "recall", "roc_auc"]].mean())
print(f"\n📁 Metrics saved to {model_dir}/")
