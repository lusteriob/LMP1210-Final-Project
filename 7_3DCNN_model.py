import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
import torch.nn.functional as F
from collections import Counter

# === Dataset Definition ===
class CWT3DImageDataset(Dataset):
    def __init__(self, data_dir, split_file=None, target_size=(64, 128)):
        self.data_dir = data_dir
        self.target_size = target_size

        if split_file:
            with open(split_file, "r") as f:
                self.patient_ids = [line.strip() for line in f if line.strip()]
        else:
            self.patient_ids = [f.split("_volume.npy")[0] for f in os.listdir(data_dir) if f.endswith("_volume.npy")]

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        volume_path = os.path.join(self.data_dir, f"{patient_id}_volume.npy")
        label_path = os.path.join(self.data_dir, f"{patient_id}_label.txt")

        volume = np.load(volume_path)  # Shape: (9, H, W)
        with open(label_path, "r") as f:
            label = int(f.read().strip())

        volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)  # (1, 9, H, W)

        # Resize to (1, 9, target_H, target_W)
        volume_tensor = F.interpolate(volume_tensor, size=(self.target_size[0], self.target_size[1]), mode='bilinear', align_corners=False)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return volume_tensor, label_tensor

# === Model Definition ===
class Simple3DCNN(nn.Module):
    def __init__(self, input_shape=(1, 9, 64, 128)):
        super(Simple3DCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )

        with torch.no_grad():
            dummy = torch.zeros((1,) + input_shape)
            dummy_out = self.feature_extractor(dummy)
            self.flattened_size = dummy_out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)

# === Config ===
DATA_DIR = "features/cwt_numpy"
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = (64, 128)

# === Load Data ===
train_dataset = CWT3DImageDataset(DATA_DIR, os.path.join(SPLIT_DIR, "train.txt"), target_size=TARGET_SIZE)
val_dataset = CWT3DImageDataset(DATA_DIR, os.path.join(SPLIT_DIR, "val.txt"), target_size=TARGET_SIZE)
test_dataset = CWT3DImageDataset(DATA_DIR, os.path.join(SPLIT_DIR, "test.txt"), target_size=TARGET_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# === Compute Class Weights ===
train_labels = []
for pid in train_dataset.patient_ids:
    with open(os.path.join(DATA_DIR, f"{pid}_label.txt")) as f:
        train_labels.append(int(f.read().strip()))

label_counts = Counter(train_labels)
total = sum(label_counts.values())
class_weights = [total / label_counts[i] for i in sorted(label_counts.keys())]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# === Initialize Model ===
sample_volume, _ = next(iter(train_loader))
input_shape = sample_volume.shape[1:]  # (1, 9, H, W)
model = Simple3DCNN(input_shape=input_shape).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Training Loop ===
best_val_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation with per-class metrics
    model.eval()
    correct = 0
    total = 0
    val_preds = []
    val_true = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    val_precision = precision_score(val_true, val_preds, average='binary', pos_label=1, zero_division=0)
    val_recall = recall_score(val_true, val_preds, average='binary', pos_label=1, zero_division=0)
    val_f1 = f1_score(val_true, val_preds, average='binary', pos_label=1, zero_division=0)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.4f} | Val Acc: {val_acc:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_3dcnn_model.pth")
        print("Saved new best model!")

# === Evaluate on Test Set ===
model.load_state_dict(torch.load("best_3dcnn_model.pth"))
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# === Final Report ===
print("\n=== Classification Report ===")
print(classification_report(all_labels, all_preds, target_names=["Healthy", "PD"]))

# === Confusion Matrix ===
print("\n=== Confusion Matrix ===")
conf_matrix = confusion_matrix(all_labels, all_preds)
print(conf_matrix)
