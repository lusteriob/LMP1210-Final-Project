import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
import numpy as np

class CWT3DImageDataset(Dataset):
    def __init__(self, data_dir, split_file=None):
        self.data_dir = data_dir

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
        label_tensor = torch.tensor(label, dtype=torch.long)

        return volume_tensor, label_tensor

#model definition
class Simple3DCNN(nn.Module):
    def __init__(self, input_shape=(1, 9, 128, 256)):  # (channels, depth, height, width)
        super(Simple3DCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

        # Auto-calculate the flattened dimension
        with torch.no_grad():
            dummy_input = torch.zeros((1,) + input_shape)
            dummy_output = self.feature_extractor(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).shape[1]

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

# === Load Data ===
train_dataset = CWT3DImageDataset(DATA_DIR, os.path.join(SPLIT_DIR, "train.txt"))
val_dataset = CWT3DImageDataset(DATA_DIR, os.path.join(SPLIT_DIR, "val.txt"))
test_dataset = CWT3DImageDataset(DATA_DIR, os.path.join(SPLIT_DIR, "test.txt"))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# === Initialize Model ===
model = Simple3DCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
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

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.4f} | Val Acc: {val_acc:.4f}")

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

print("\n=== Classification Report ===")
print(classification_report(all_labels, all_preds, target_names=["Healthy", "PD"]))
