import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import tensorflow as tf
from keras import layers, models, callbacks
from sklearn.metrics import balanced_accuracy_score


# === Load data ===
DATA_DIR = "all_processed_data"
BIN_META_PATH = os.path.join(DATA_DIR, "bin_metadata.csv")
NUM_SEGMENTS = 66
SEGMENT_LENGTH = 976
INPUT_SHAPE = (SEGMENT_LENGTH, NUM_SEGMENTS)

# Load bin file
def load_bin_file(path):
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(NUM_SEGMENTS, SEGMENT_LENGTH).T  # (976, 66)

# Load all data
bin_metadata = pd.read_csv(BIN_META_PATH)
X_list, y_list = [], []

for _, row in tqdm(bin_metadata.iterrows(), total=len(bin_metadata), desc="Loading time-series"):
    X_list.append(load_bin_file(row["FilePath"]))
    y_list.append(row["Label"])

X = np.stack(X_list)
y = np.array(y_list)

# === Model builder ===
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

# === Hyperparameters to tune
param_grid = [
    {"filters": 64, "kernel_size": 5, "dropout": 0.5},
    {"filters": 128, "kernel_size": 3, "dropout": 0.3},
    {"filters": 64, "kernel_size": 7, "dropout": 0.5},
]

# === Nested CV Setup
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

results = []

# === Outer CV loop
for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    print(f"\n🔁 Outer Fold {fold_idx + 1}/5")

    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]

    best_score = -1
    best_params = None

    # === Inner loop: Hyperparameter tuning
    for params in param_grid:
        val_scores = []

        for inner_train_idx, val_idx in inner_cv.split(X_train_outer, y_train_outer):
            X_train_inner = X_train_outer[inner_train_idx]
            y_train_inner = y_train_outer[inner_train_idx]
            X_val = X_train_outer[val_idx]
            y_val = y_train_outer[val_idx]

            model = build_model(**params)

            early_stop = callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )

            model.fit(
                X_train_inner, y_train_inner,
                class_weight="balanced",
                validation_data=(X_val, y_val),
                epochs=50, batch_size=32,
                callbacks=[early_stop], verbose=0
            )

            y_val_pred = model.predict(X_val).ravel()
            auc = roc_auc_score(y_val, y_val_pred)
            val_scores.append(auc)

        avg_val_score = np.mean(val_scores)
        print(f"   🔍 Params {params} → Avg AUC: {avg_val_score:.4f}")
        if avg_val_score > best_score:
            best_score = avg_val_score
            best_params = params

    # === Final training with best params
    print(f"🧠 Best inner params: {best_params}")
    final_model = build_model(**best_params)

    early_stop_outer = callbacks.EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True
    )

    history = final_model.fit(
        X_train_outer, y_train_outer,
        class_weight="balanced",
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop_outer],
        verbose=0
    )

    model_path = f"models/cnn_models/cnn_fold{fold_idx + 1}_best.h5"
    final_model.save(model_path)

    y_test_pred = final_model.predict(X_test_outer).ravel()
    test_auc = roc_auc_score(y_test_outer, y_test_pred)
    test_acc = accuracy_score(y_test_outer, y_test_pred > 0.5)
    test_bacc = balanced_accuracy_score(y_test_outer, y_test_pred > 0.5)

    print(f"✅ Fold {fold_idx + 1}: AUC={test_auc:.4f}, ACC={test_acc:.4f}")

    results.append({
        "fold": fold_idx + 1,
        "auc": test_auc,
        "acc": test_acc,
        "bacc": test_bacc,
        "params": best_params
    })

# === Save + Display results
results_df = pd.DataFrame(results)
results_df.to_csv("models/cnn_models/cnn_nestedcv_results.csv", index=False)

print("\n📊 Nested CV Summary:")
print(results_df)
print("\n✅ Mean AUC:", results_df['auc'].mean())
print("✅ Mean ACC:", results_df['acc'].mean())
print("✅ Mean BACC:", results_df['bacc'].mean())
