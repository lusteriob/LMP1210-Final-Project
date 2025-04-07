import os

data_dir = "features/cwt_numpy"
npy_files = [f for f in os.listdir(data_dir) if f.endswith("_volume.npy")]
print(f"Total .npy volume files: {len(npy_files)}")
split_dir = "features/cwt_numpy/splits"

for split in ["train.txt", "val.txt", "test.txt"]:
    with open(os.path.join(split_dir, split)) as f:
        lines = f.readlines()
        print(f"{split}: {len(lines)} patients")
