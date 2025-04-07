import os
import random

# Set seed for reproducibility
random.seed(42)

# === Paths ===
data_dir = "features/cwt_numpy"
split_dir = os.path.join(data_dir, "splits")
os.makedirs(split_dir, exist_ok=True)

# === Get all patient IDs from saved .npy files ===
patient_ids = [
    fname.split("_volume.npy")[0]
    for fname in os.listdir(data_dir)
    if fname.endswith("_volume.npy")
]

# Shuffle and split
random.shuffle(patient_ids)
n_total = len(patient_ids)
n_train = int(0.6 * n_total)
n_val = int(0.2 * n_total)

train_ids = patient_ids[:n_train]
val_ids = patient_ids[n_train:n_train + n_val]
test_ids = patient_ids[n_train + n_val:]

# === Save splits ===
def save_split(ids, filename):
    with open(os.path.join(split_dir, filename), "w") as f:
        for pid in ids:
            f.write(f"{pid}\n")

save_split(train_ids, "train.txt")
save_split(val_ids, "val.txt")
save_split(test_ids, "test.txt")

print("Train/Val/Test splits saved!")
print(f"Train: {len(train_ids)} samples")
print(f"Val:   {len(val_ids)} samples")
print(f"Test:  {len(test_ids)} samples")


import os
import torch
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
         # Normalize volume to [0, 1] range
        volume = volume / 255.0
        with open(label_path, "r") as f:
            label = int(f.read().strip())

        volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 9, H, W)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return volume_tensor, label_tensor