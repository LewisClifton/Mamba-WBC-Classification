import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, ConcatDataset

from medmnist import BloodMNIST


# Dataset for the generated augmented images
class AugBloodMNIST(Dataset):
    def __init__(self, images_path, labels_path):
        super(AugBloodMNIST, self).__init__()
        self.images_path = images_path
        self.labels = pd.read_csv(labels_path)
        self.images = os.listdir(images_path)

    def __getitem__(self, idx):

        # Get augmented image and label
        row = self.labels.iloc[idx]
        image_path = os.path.join(self.images_path, row['image_name'])
        image = Image.open(image_path).convert("RGB")

        label = row['label']

        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        
        return image, label, None

    def __len__(self):
        return len(self.images)


# Dataset wrapper to allows concatenation with the augmented train set
class BloodMNISTWrapper(Dataset):
    def __init__(self, dataset_download_dir, split='train'):
        super(BloodMNISTWrapper, self).__init__()
        self.dataset = BloodMNIST(split=split, download=True, size=224, root=dataset_download_dir)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, torch.tensor(label, dtype=torch.float32), None

    def __len__(self):
        return len(self.dataset)


# Dataset wrapper which combines the pre-defined test set with the generated augmented images
class TrainBloodMNIST(Dataset):
    def __init__(self, images_path, labels_path, dataset_download_dir):
        super(TrainBloodMNIST, self).__init__()
        train_dataset = BloodMNISTWrapper(dataset_download_dir, split='train')

        aug_dataset = AugBloodMNIST(images_path, labels_path)
        
        # Combine the original train set with the augmented dataset
        self.dataset = ConcatDataset([train_dataset, aug_dataset])


    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
    

def get(dataset_config, dataset_download_dir, test=False, augmented_data_dir=None):

    # Set up path for the dataset 
    if not os.path.isdir(dataset_download_dir):
        os.mkdir(dataset_download_dir)

    # Get the train and validation set or test set
    if not test:
        # Use the augmented train set if necessary
        if 'augmented_images_dir' in dataset_config.keys():
            train_dataset = TrainBloodMNIST(dataset_config['augmented_images_dir'], dataset_config['augmented_labels_path'], dataset_download_dir)
        else:
            train_dataset = BloodMNISTWrapper(dataset_download_dir, split='train')

        val = BloodMNISTWrapper(dataset_download_dir, split='val')
        return train_dataset, val_dataset
    else: 
        test_dataset = BloodMNISTWrapper(dataset_download_dir, split='test')
        return test_dataset