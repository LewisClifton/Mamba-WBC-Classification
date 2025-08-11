import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset, Image as HFImage
from datasets.download.download_config import DownloadConfig
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
from pathlib import Path

class ChulaWBC5000(Dataset):
    def __init__(self, hf_dataset, wbc_types, test=False):
        super().__init__()
        self.dataset = hf_dataset.filter(lambda x: x['label'] in wbc_types)
        self.wbc_types = wbc_types
        self.wbc_type_to_class = {label: idx for idx, label in enumerate(wbc_types)}
        self.class_to_wbc_type = {idx: label for idx, label in enumerate(wbc_types)}
        self.test = test

    def get_wbc_type_from_class(self, class_):
        return self.class_to_wbc_type[class_]

    def get_class_from_wbc_type(self, label):
        return self.wbc_type_to_class[label]

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        label_idx = self.get_class_from_wbc_type(item['label'])
        label = torch.tensor(label_idx, dtype=torch.long).unsqueeze(0)
        return image, label, item['name']

    def __len__(self):
        return len(self.dataset)


def get(dataset_config, test=False):
    repo_id = dataset_config['huggingface_url']  # e.g. "LewisClifton/Chula-WBC-8"

    # Load token from environment variable
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token is None:
        raise EnvironmentError("HUGGINGFACE_TOKEN environment variable not set")

    print('Downloading dataset files...')

    # Use hf_hub_download to get the local path to the entire dataset snapshot
    # This downloads all files (labels.csv and images in subdirectories)
    # to the Hugging Face cache.
    local_dataset_path = hf_hub_download(
        repo_id=repo_id,
        filename="labels.csv", # Specify one file to trigger the snapshot download
        token=token,
        repo_type="dataset",
        local_dir_use_symlinks=False # Set to False for direct files, True for symlinks
    )
    # The `hf_hub_download` returns the path to the specific file,
    # we need the directory where this file and other assets are
    # The parent directory is where the snapshot lives.
    # If using local_dir_use_symlinks=True, the parent is the cached repo root.
    # If using local_dir_use_symlinks=False, the parent is where filename itself is.
    # For a dataset with structure like repo/labels.csv, repo/images/..., the local_dataset_path
    # will be something like ~/.cache/huggingface/hub/datasets/LewisClifton___Chula-WBC-8/.../labels.csv
    # So we need its parent directory.
    dataset_base_dir = Path(local_dataset_path).parent

    print(f'Dataset files downloaded to: {dataset_base_dir}')

    # Load the CSV file into a DataFrame
    csv_file_path = dataset_base_dir / "labels.csv"
    if not csv_file_path.exists():
        raise FileNotFoundError(f"labels.csv not found at {csv_file_path}. Check repo structure.")

    df = pd.read_csv(csv_file_path)

    # Construct the full image paths
    # Assuming image filenames are in a column named 'name' or 'image_name' etc.
    # Based on your __getitem__ accessing item['name'], let's assume 'name'
    if 'name' not in df.columns:
        raise ValueError("DataFrame must contain an 'name' column for image filenames.")
    
    # Prepend the base directory and 'images/' to the image filenames
    df['image'] = df['name'].apply(lambda filename: str(dataset_base_dir / "images" / filename))

    # Create Hugging Face Dataset from DataFrame
    # And cast the 'image' column to an Image feature
    hf_dataset = HFDataset.from_pandas(df).cast_column("image", HFImage())

    # Get sorted list of unique WBC labels
    # Make sure your 'labels.csv' has a 'label' column for this.
    if 'label' not in df.columns:
        raise ValueError("DataFrame must contain a 'label' column for classification.")
    wbc_types = sorted(df["label"].unique().tolist())

    print('Hugging Face Dataset created.')

    return ChulaWBC5000(hf_dataset, wbc_types=wbc_types, test=test)
