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