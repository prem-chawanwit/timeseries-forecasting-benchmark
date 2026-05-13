import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("joebeachcapital/metropt-3-dataset")

print("Path to dataset files:", path)

# Copy the downloaded dataset to our data/1_raw folder for consistency
raw_data_dir = "data/1_raw/metropt-3-dataset"
if not os.path.exists(raw_data_dir):
    shutil.copytree(path, raw_data_dir)
    print(f"Dataset copied to {raw_data_dir}")
else:
    print(f"Dataset already exists in {raw_data_dir}")
