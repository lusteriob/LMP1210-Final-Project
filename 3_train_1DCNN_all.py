import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
import tensorflow as tf
from keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight

# === BACC EarlyStopping Callback ===
class BalancedAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, patience=5):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.patience = patience
        self.best_bacc = 0
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val).ravel()
        y_pred_labels = (y_pred > 0.5).astype(int)
        bacc = balanced_accuracy_score(self.y_val, y_pred_labels)
        print(f"🔎 Epoch {epoch+1}: BACC={bacc:.4f} (Best: {self.best_bacc:.4f})")

        if bacc > self.best_bacc:
            self.best_bacc = bacc
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print("⏹️ Early stopping triggered by BACC")
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

# === Paths and Constants ===
DATA_DIR = "all_processed_data"
BIN_META_PATH = os.path.join(DATA_DIR, "bin_metadata.csv")
MODEL_DIR = "models/cnn_models"
os.makedirs(MODEL_DIR, exist_ok=True)

NUM_SEGMENTS = 66
SEGMENT_LENGTH = 976
INPUT_SHAPE = (SEGMENT_LENGTH, NUM_SEGMENTS)

# === Load Data ===
def load_bin_file(path):
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(NUM_SEGMENTS, SEGMENT_LENGTH).T

bin_metadata = pd.read_csv(BIN_META_PATH)
X_list, y_list = [], []
for _, row in tqdm(bin_metadata.iterrows(), total=len(bin_metadata), desc="Loading time-series"):
    X_list.append(load_bin_file(row["FilePath"]))
    y_list.append(row["Label"])
X = np.stack(X_list)
y = np.array(y_list)

# === Build CNN Model ===
def build_model(filters=64, kernel_size=5, dropout=0.5):
    model = models.Sequential([
        layers.Conv1D(filters, kernel_size=kernel_size, activation='relu', input_shape=INPUT_SHAPE),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters * 2, kernel_size=kernel_size, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    return model

# === Hyperparameter Grid ===
param_grid = [
    {"filters": f, "kernel_size": k, "dropout": d}
    for f, k, d in product([64, 128], [3, 5, 7], [0.3, 0.4, 0.5])
]

# === Nested CV Setup ===
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    print(f"\n🔁 Outer Fold {fold_idx + 1}/5")
    X_outer_train, X_outer_test = X[train_idx], X[test_idx]
    y_outer_train, y_outer_test = y[train_idx], y[test_idx]

    best_inner_score = -1
    best_model = None
    best_params = None

    for params in param_grid:
        inner_scores = []
        inner_models = []

        for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_outer_train, y_outer_train)):
            print(f"     🧪 Inner Fold {inner_fold_idx + 1}/5 for params {params}")
            X_inner_train = X_outer_train[inner_train_idx]
            y_inner_train = y_outer_train[inner_train_idx]
            X_inner_val = X_outer_train[inner_val_idx]
            y_inner_val = y_outer_train[inner_val_idx]

            model = build_model(**params)
            bacc_cb = BalancedAccuracyCallback(validation_data=(X_inner_val, y_inner_val), patience=5)

            weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_inner_train), y=y_inner_train)
            class_weights_dict = dict(zip(np.unique(y_inner_train), weights))

            model.fit(
                X_inner_train, y_inner_train,
                class_weight=class_weights_dict,
                validation_data=(X_inner_val, y_inner_val),
                epochs=50, batch_size=32,
                callbacks=[bacc_cb], verbose=0
            )

            y_pred = model.predict(X_inner_val).ravel()
            bacc = balanced_accuracy_score(y_inner_val, y_pred > 0.5)
            inner_scores.append(bacc)
            inner_models.append(model)

        avg_bacc = np.mean(inner_scores)
        print(f"   🔍 Params {params} → Avg BACC: {avg_bacc:.4f}")

        if avg_bacc > best_inner_score:
            best_inner_score = avg_bacc
            best_model = inner_models[np.argmax(inner_scores)]  # Best model per inner CV
            best_params = params

    # === Evaluate best model on outer fold ===
    y_test_pred = best_model.predict(X_outer_test).ravel()
    test_auc = roc_auc_score(y_outer_test, y_test_pred)
    test_acc = accuracy_score(y_outer_test, y_test_pred > 0.5)
    test_bacc = balanced_accuracy_score(y_outer_test, y_test_pred > 0.5)

    print(f"✅ Fold {fold_idx + 1}: AUC={test_auc:.4f}, ACC={test_acc:.4f}, BACC={test_bacc:.4f}")

    # Save model
    best_model.save(os.path.join(MODEL_DIR, f"cnn_fold{fold_idx + 1}_best_v2.h5"))

    results.append({
        "fold": fold_idx + 1,
        "auc": test_auc,
        "acc": test_acc,
        "bacc": test_bacc,
        "params": best_params
    })

# === Save Results ===
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(MODEL_DIR, "cnn_nestedcv_results_v2.csv"), index=False)

print("\n📊 Nested CV Summary:")
print(results_df)
print("\n✅ Mean AUC:", results_df['auc'].mean())
print("✅ Mean ACC:", results_df['acc'].mean())
print("✅ Mean BACC:", results_df['bacc'].mean())