import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
from tqdm import tqdm
import random

from monai.networks.nets import resnet

# === Dataset Definition ===
class CWT3DImageDataset(Dataset):
    def __init__(self, data_dir, patient_ids, target_size=(16, 96, 96), augment_healthy=False):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.target_size = target_size
        self.augment_healthy = augment_healthy

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        volume = np.load(os.path.join(self.data_dir, f"{patient_id}_volume.npy"))

        if volume.ndim != 3:
            raise ValueError(f"Expected shape (D, H, W) but got {volume.shape}")

        with open(os.path.join(self.data_dir, f"{patient_id}_label.txt")) as f:
            label = int(f.read().strip())

        # Gaussian noise only for healthy class
        if self.augment_healthy and label == 0 and np.random.rand() < 0.5:
            noise = np.random.normal(0.0, 0.01, volume.shape)
            volume = np.clip(volume + noise, 0.0, 1.0)

        volume_tensor = torch.tensor(volume, dtype=torch.float32)
        volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)
        volume_tensor = F.interpolate(volume_tensor, size=self.target_size, mode='trilinear', align_corners=False)
        volume_tensor = volume_tensor.squeeze(0)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return volume_tensor, label_tensor

# === Training Function ===
def train_model(model, train_loader, val_loader, class_weights, lr, device, epochs=50, patience=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': lr * 0.1},
        {'params': model.fc.parameters(), 'lr': lr}
    ])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

    best_score = 0.0
    no_improve_epochs = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        val_acc = np.mean(np.array(val_preds) == np.array(val_true))
        val_precision = precision_score(val_true, val_preds, average='macro', zero_division=0)
        val_recall = recall_score(val_true, val_preds, average='macro', zero_division=0)
        val_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)
        val_bacc = balanced_accuracy_score(val_true, val_preds)

        recall_per_class = recall_score(val_true, val_preds, average=None, zero_division=0)
        healthy_recall = recall_per_class[0] if len(recall_per_class) > 1 else 0.0

        scheduler.step(val_bacc)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f} | Val Acc: {val_acc:.4f} | Macro Precision: {val_precision:.4f} | Macro Recall: {val_recall:.4f} | Macro F1: {val_f1:.4f} | Balanced Acc: {val_bacc:.4f}")
        print(f"  Healthy Recall: {healthy_recall:.4f}")

        if val_bacc > best_score and healthy_recall >= 0.3:
            best_score = val_bacc
            no_improve_epochs = 0
            best_model_state = model.state_dict()
            print("Saved new best model!")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epoch(s).")
            if no_improve_epochs >= patience:
                print("Early stopping triggered.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

# === Evaluation Function ===
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=["Healthy", "PD"]))
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(all_labels, all_preds))

# === Main Training Driver ===
def run_resnet_with_aug(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_ids = [f.split("_volume.npy")[0] for f in os.listdir(data_dir) if f.endswith("_volume.npy")]
    all_labels = [int(open(os.path.join(data_dir, f"{pid}_label.txt")).read().strip()) for pid in all_ids]

    batch_size = 8
    target_size = (16, 96, 96)
    lr = 1e-3

    train_ids, test_ids, train_labels, test_labels = train_test_split(all_ids, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.25, stratify=train_labels, random_state=42)

    train_ds = CWT3DImageDataset(data_dir, train_ids, target_size=target_size, augment_healthy=True)
    val_ds = CWT3DImageDataset(data_dir, val_ids, target_size=target_size)
    test_ds = CWT3DImageDataset(data_dir, test_ids, target_size=target_size)

    label_counts = Counter([int(open(os.path.join(data_dir, f"{pid}_label.txt")).read()) for pid in train_ids])
    sample_weights = [1.0 / label_counts[int(open(os.path.join(data_dir, f"{pid}_label.txt")).read())] for pid in train_ids]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = resnet.resnet18(spatial_dims=3, n_input_channels=1, num_classes=2, pretrained=False)
    model = model.to(device)

    class_weights = None  # unweighted loss
    model = train_model(model, train_loader, val_loader, class_weights, lr, device)

    print("\n--- Evaluating Final Model on Test Set ---")
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    DATA_DIR = "features/cwt_numpy"
    run_resnet_with_aug(DATA_DIR)
