import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
from tqdm import tqdm

# === Dataset Definition ===
class CWT3DImageDataset(Dataset):
    def __init__(self, data_dir, patient_ids, target_size=(64, 128)):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.target_size = target_size

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        volume = np.load(os.path.join(self.data_dir, f"{patient_id}_volume.npy"))
        with open(os.path.join(self.data_dir, f"{patient_id}_label.txt")) as f:
            label = int(f.read().strip())

        volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)
        volume_tensor = F.interpolate(volume_tensor, size=self.target_size, mode='bilinear', align_corners=False)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return volume_tensor, label_tensor

# === Enhanced Model Definition ===
class Enhanced3DCNN(nn.Module):
    def __init__(self, input_shape=(1, 9, 64, 128)):
        super(Enhanced3DCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

        with torch.no_grad():
            dummy = torch.zeros((1,) + input_shape)
            dummy_out = self.feature_extractor(dummy)
            self.flattened_size = dummy_out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)

# === Training Function ===
def train_model(model, train_loader, val_loader, class_weights, lr, device, epochs=50, patience=3):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

    best_val_f1 = 0.0
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

        scheduler.step(val_f1)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f} | Val Acc: {val_acc:.4f} | Macro Precision: {val_precision:.4f} | Macro Recall: {val_recall:.4f} | Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
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
def run_simple_training(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_ids = [f.split("_volume.npy")[0] for f in os.listdir(data_dir) if f.endswith("_volume.npy")]
    all_labels = [int(open(os.path.join(data_dir, f"{pid}_label.txt")).read().strip()) for pid in all_ids]

    batch_size = 8
    target_size = (64, 128)
    lr = 1e-3

    train_ids, test_ids, train_labels, test_labels = train_test_split(all_ids, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.25, stratify=train_labels, random_state=42)

    train_ds = CWT3DImageDataset(data_dir, train_ids, target_size=target_size)
    val_ds = CWT3DImageDataset(data_dir, val_ids, target_size=target_size)
    test_ds = CWT3DImageDataset(data_dir, test_ids, target_size=target_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    label_counts = Counter([int(open(os.path.join(data_dir, f"{pid}_label.txt")).read()) for pid in train_ids])
    total = sum(label_counts.values())
    weights = torch.tensor([total / label_counts[i] for i in sorted(label_counts.keys())], dtype=torch.float32).to(device)

    sample_input = next(iter(train_loader))[0]
    input_shape = sample_input.shape[1:]
    model = Enhanced3DCNN(input_shape=input_shape).to(device)
    model = train_model(model, train_loader, val_loader, weights, lr, device)

    print("\n--- Evaluating Final Model on Test Set ---")
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    DATA_DIR = "features/cwt_numpy"
    run_simple_training(DATA_DIR)
