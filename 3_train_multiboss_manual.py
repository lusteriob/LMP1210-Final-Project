import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, make_scorer
)
import warnings
warnings.filterwarnings("ignore")

# === Load features ===
def load_feature_matrix(name):
    if name == "multiboss":
        df = pd.read_pickle("features/multiboss_features.pkl")
    else:
        df = pd.read_pickle(f"features/{name}_features_pads_style.pkl")
    if "PatientID" in df.columns:
        df = df.drop(columns=["PatientID"])
    return df.astype(np.float32).values

X_manual = load_feature_matrix("manual")
X_multiboss = load_feature_matrix("multiboss")
y = pd.read_csv("features/manual_labels_pads_style.csv")["Label"]

# === Build feature set combinations ===
feature_sets = {
    "manual": X_manual,
    "multiboss": X_multiboss,
    "manual+multiboss": np.hstack([X_manual, X_multiboss])
}

# === Identity transformer ===
class Identity(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

# === Output dir ===
model_dir = "models/svm_nested_cv_multiboss"
os.makedirs(model_dir, exist_ok=True)

# === CV Setup ===
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === Results storage ===
results = []

print("🔁 Running 5x5 nested CV on MultiBOSS feature sets...\n")

for name, X in feature_sets.items():
    print(f"🔍 Feature Set: {name}")

    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(outer_cv.split(X, y), total=5, desc=f"{name} folds")):
        print(f"\n🔄 Fold {fold_idx + 1}/5")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = Pipeline([
            ("reduce", Identity()),
            ("scaler", StandardScaler()),
            ("svm", SVC(class_weight="balanced", probability=True))
        ])

        param_grid = {
            "reduce": [
                Identity(),
                PCA(n_components=100),
                PCA(n_components=300),
                SelectKBest(f_classif, k=200),
                SelectKBest(f_classif, k=500)
            ],
            "svm__C": [0.1, 1, 10],
            "svm__gamma": ["scale", 0.01],
            "svm__kernel": ["rbf"]
        }

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

        reducer = grid.best_params_["reduce"]
        reducer_name = reducer.__class__.__name__

        model_path = os.path.join(model_dir, f"svm_{name}_fold{fold_idx + 1}.joblib")
        joblib.dump(best_model, model_path)

        metrics = {
            "feature_set": name,
            "fold": fold_idx + 1,
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "best_params": grid.best_params_,
            "reducer": reducer_name
        }
        results.append(metrics)

        print(f"✅ Fold {metrics['fold']} done: "
              f"BA={metrics['balanced_accuracy']:.4f}, "
              f"F1={metrics['f1']:.4f}, "
              f"AUC={metrics['roc_auc']:.4f}")
        print(f"   Reducer: {reducer_name}")
        print(f"   Best Params: {metrics['best_params']}\n")

# === Save Results ===
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(model_dir, "metrics_summary.csv"), index=False)

print("\n✅ Finished nested CV on MultiBOSS feature sets.")
print("📊 Average performance by feature set:")
print(results_df.groupby("feature_set")[["balanced_accuracy", "f1", "precision", "recall", "roc_auc"]].mean())
