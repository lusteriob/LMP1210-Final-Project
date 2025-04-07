import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import gc

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, f1_score

# ========== CONFIG ========== #
DATA_DIR = "features_all/cwt_numpy_volumes"
OUTPUT_DIR = "cnn_nestedcv_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
NUM_OUTER_FOLDS = 5
NUM_INNER_FOLDS = 3
EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 4
TARGET_SIZE = (64, 128)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================ #

# === Dataset with Optional Gaussian Noise === #
class CWT3DImageDataset(Dataset):
    def __init__(self, data_dir, patient_ids, target_size=(64, 128), noise_std=0.0):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.target_size = target_size
        self.noise_std = noise_std

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        volume = np.load(os.path.join(self.data_dir, f"{patient_id}_volume.npy"))
        with open(os.path.join(self.data_dir, f"{patient_id}_label.txt")) as f:
            label = int(f.read().strip())

        volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)
        volume_tensor = F.interpolate(volume_tensor, size=self.target_size, mode='bilinear', align_corners=False)

        if self.noise_std > 0:
            noise = torch.randn_like(volume_tensor) * self.noise_std
            volume_tensor += noise

        label_tensor = torch.tensor(label, dtype=torch.long)
        return volume_tensor, label_tensor

# === Model Definition === #
class Enhanced3DCNN(nn.Module):
    def __init__(self, input_shape, filters=16, kernel_size=3, dropout=0.5):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(1, filters, kernel_size=kernel_size, padding=1),
            nn.BatchNorm3d(filters),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(filters, filters * 2, kernel_size=kernel_size, padding=1),
            nn.BatchNorm3d(filters * 2),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(filters * 2, filters * 4, kernel_size=kernel_size, padding=1),
            nn.BatchNorm3d(filters * 4),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2))
        )

        with torch.no_grad():
            dummy = torch.zeros((1,) + input_shape)
            dummy_out = self.feature_extractor(dummy)
            flattened = dummy_out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))

# === Training Loop === #
def train_model(model, train_loader, val_loader, lr, device, patience=PATIENCE, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=False)

    best_model_state = None
    best_bacc = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                output = model(x)
                _, predicted = torch.max(output, 1)
                preds.extend(predicted.cpu().numpy())
                labels.extend(y.numpy())

        bacc = balanced_accuracy_score(labels, preds)
        scheduler.step(bacc)

        if bacc > best_bacc:
            best_bacc = bacc
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

        torch.cuda.empty_cache()
        gc.collect()

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

# === Evaluation Function === #
def evaluate_model(model, loader):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            output = model(x)
            probs_batch = F.softmax(output, dim=1)[:, 1].cpu().numpy()
            _, predicted = torch.max(output, 1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(y.numpy())
            probs.extend(probs_batch)

    return {
        "acc": accuracy_score(labels, preds),
        "bacc": balanced_accuracy_score(labels, preds),
        "auc": roc_auc_score(labels, probs),
        "f1": f1_score(labels, preds)
    }

# === Task Importance with LOTO ablation === #

def mask_task(tensor_batch, task_idx, total_tasks=11, channels_per_task=6):
    """
    Zero out all channels associated with a given task.
    """
    tensor_batch = tensor_batch.clone()  # (B, C=1, D=66, H, W)
    expected_channels = total_tasks * channels_per_task
    if tensor_batch.shape[2] != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels in depth, got {tensor_batch.shape[2]}")

    start = task_idx * channels_per_task
    end = start + channels_per_task
    tensor_batch[:, :, start:end] = 0.0
    return tensor_batch

def task_ablation(model, test_loader, num_tasks=11, channels_per_task=6):
    print("\n📉 Starting Leave-One-Task-Out (LOTO) Ablation...")
    full_metrics = evaluate_model(model, test_loader)
    baseline_bacc = full_metrics["bacc"]
    print(f"Baseline BACC: {baseline_bacc:.4f}")

    drops = []
    for task_idx in range(num_tasks):
        preds, labels, probs = [], [], []
        with torch.no_grad():
            for x, y in test_loader:
                x_masked = mask_task(x, task_idx, total_tasks=num_tasks, channels_per_task=channels_per_task)
                x_masked = x_masked.to(DEVICE, non_blocking=True)
                output = model(x_masked)
                probs_batch = F.softmax(output, dim=1)[:, 1].cpu().numpy()
                _, predicted = torch.max(output, 1)
                preds.extend(predicted.cpu().numpy())
                labels.extend(y.numpy())
                probs.extend(probs_batch)
        bacc_masked = balanced_accuracy_score(labels, preds)
        drop = baseline_bacc - bacc_masked
        print(f"Task {task_idx} ablation: ΔBACC = {drop:.4f}")
        drops.append((task_idx, drop))

    print("\n📊 Task Importance Ranking (most impactful first):")
    for i, (task, drop) in enumerate(sorted(drops, key=lambda x: -x[1])):
        print(f"{i+1}. Task {task} → ΔBACC = {drop:.4f}")

    return drops


# === Main Nested CV Logic === #
def nested_cv():
    all_ids = [f.split("_volume.npy")[0] for f in os.listdir(DATA_DIR) if f.endswith("_volume.npy")]
    all_labels = [int(open(os.path.join(DATA_DIR, f"{pid}_label.txt")).read().strip()) for pid in all_ids]

    outer_cv = StratifiedKFold(n_splits=NUM_OUTER_FOLDS, shuffle=True, random_state=42)
    results = []

    # Grid parameters
    grid = list(product([16, 32], [3], [0.3, 0.5], [0.0, 0.1]))  # filters, kernel, dropout, noise_std

    print("\n🔁 Starting Outer Fold Loop...")
    for outer_idx, (trainval_idx, test_idx) in tqdm(list(enumerate(outer_cv.split(all_ids, all_labels))),
                                                    desc="🔁 Outer Folds", total=NUM_OUTER_FOLDS):
        trainval_ids = [all_ids[i] for i in trainval_idx]
        trainval_labels = [all_labels[i] for i in trainval_idx]
        test_ids = [all_ids[i] for i in test_idx]
        test_labels = [all_labels[i] for i in test_idx]

        print(f"\n🧪 Outer Fold {outer_idx + 1}/{NUM_OUTER_FOLDS} - Train/Val: {len(trainval_ids)}, Test: {len(test_ids)}")
        print("🔍 Starting Grid Search over", len(grid), "configs...")

        best_score, best_model, best_params = -1, None, None
        inner_cv = StratifiedKFold(n_splits=NUM_INNER_FOLDS, shuffle=True, random_state=42)

        for filters, ksize, dropout, noise_std in tqdm(grid, desc="🔍 Grid Search (Inner)", leave=False):
            print(f"  🧪 Trying config: filters={filters}, kernel_size={ksize}, dropout={dropout}, noise_std={noise_std}")
            fold_scores = []

            for inner_train_idx, val_idx in tqdm(list(inner_cv.split(trainval_ids, trainval_labels)),
                                                 desc="📊 Inner Folds", leave=False):
                inner_train_ids = [trainval_ids[i] for i in inner_train_idx]
                inner_val_ids = [trainval_ids[i] for i in val_idx]

                train_ds = CWT3DImageDataset(DATA_DIR, inner_train_ids, TARGET_SIZE, noise_std=noise_std)
                val_ds = CWT3DImageDataset(DATA_DIR, inner_val_ids, TARGET_SIZE, noise_std=0.0)
                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
                val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=True)

                sample_input = next(iter(train_loader))[0]
                model = Enhanced3DCNN(input_shape=sample_input.shape[1:], filters=filters,
                                      kernel_size=ksize, dropout=dropout).to(DEVICE)
                print("    🚂 Training model...")
                model = train_model(model, train_loader, val_loader, lr=1e-3, device=DEVICE)
                metrics = evaluate_model(model, val_loader)
                print(f"    📊 Inner Fold BACC: {metrics['bacc']:.4f}, F1: {metrics['f1']:.4f}")
                fold_scores.append(metrics["bacc"])

            avg_bacc = np.mean(fold_scores)
            print(f"  ✅ Config average BACC: {avg_bacc:.4f}")
            if avg_bacc > best_score:
                best_score = avg_bacc
                best_params = (filters, ksize, dropout, noise_std)
                best_model = model

        print("🧪 Evaluating best model on outer test set...")
        test_ds = CWT3DImageDataset(DATA_DIR, test_ids, TARGET_SIZE)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, pin_memory=True)
        final_metrics = evaluate_model(best_model, test_loader)

        # LOTO ablation + save results
        drops = task_ablation(best_model, test_loader, num_tasks=11, channels_per_task=6)
        df_importance = pd.DataFrame(drops, columns=["task", "delta_bacc"])
        df_importance.to_csv(os.path.join(OUTPUT_DIR, f"fold{outer_idx+1}_task_importance.csv"), index=False)

        print(f"\n✅ Fold {outer_idx + 1}: Best Params={best_params}, BACC={final_metrics['bacc']:.4f}")
        torch.save(best_model.state_dict(), os.path.join(OUTPUT_DIR, f"cnn_fold{outer_idx+1}_best.pt"))

        results.append({
            "fold": outer_idx + 1,
            "filters": best_params[0],
            "kernel_size": best_params[1],
            "dropout": best_params[2],
            "noise_std": best_params[3],
            **final_metrics
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "nestedcv_results.csv"), index=False)
    print("\n📊 Nested CV Results Summary:")
    print(df)
    print(f"\n✅ Mean BACC: {df['bacc'].mean():.4f}")

if __name__ == "__main__":
    nested_cv()
