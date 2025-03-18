import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, ConcatDataset

from medmnist import BloodMNIST

class AugBloodMNIST(Dataset):
    def __init__(self, images_path, labels_path):
        super(AugBloodMNIST, self).__init__()
        self.images_path = images_path
        self.labels = pd.read_csv(labels_path)

        self.images = os.listdir(images_path)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        image_path = os.path.join(self.images_path, row['image_name'])
        image = Image.open(image_path).convert("RGB")

        label = row['label']

        label = torch.tensor(label, dtype=torch.long).unsqueeze(0)

        
        return image, label

    def __len__(self):
        return len(self.images)


class BloodMNISTWrapper(Dataset):
    def __init__(self, dataset_download_dir):
        super(BloodMNISTWrapper, self).__init__()
        self.dataset = BloodMNIST(split='train', download=True, size=224, root=dataset_download_dir)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)


class TrainBloodMNIST(Dataset):
    def __init__(self, images_path, labels_path, dataset_download_dir):
        super(TrainBloodMNIST, self).__init__()
        train_dataset = BloodMNISTWrapper(dataset_download_dir)

        aug_dataset = AugBloodMNIST(images_path, labels_path)
        
        self.dataset = ConcatDataset([train_dataset, aug_dataset])


    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
    

def get(dataset_config, dataset_download_dir, test=False, augmented_data_dir=None):

    # Set up path for the dataset 
    if dataset_download_dir:
        if not os.path.isdir(dataset_download_dir):
            os.mkdir(dataset_download_dir)

        # Get the dataset
        if not test:
            if 'augmented_images_dir' in dataset_config.keys():
                train_dataset = TrainBloodMNIST(dataset_config['augmented_images_dir'], dataset_config['augmented_labels_path'], dataset_download_dir)
            else:
                train_dataset = BloodMNIST(split='train', download=True, size=224, root=dataset_download_dir)

            val_dataset = BloodMNIST(split='val', download=True, size=224, root=dataset_download_dir)
            return train_dataset, val_dataset
        else: 
            test_dataset = BloodMNIST(split='test', download=True, size=224, root=dataset_download_dir)
            return test_dataset
        
    else:

        # Get the dataset
        if not test:
            if 'augmented_images_dir' in dataset_config.keys():
                train_dataset = TrainBloodMNIST(dataset_config['augmented_images_dir'], dataset_config['augmented_labels_path'], dataset_download_dir)
            else:
                train_dataset = BloodMNIST(split='train', download=True, size=224)

            val_dataset = BloodMNIST(split='val', download=True, size=224)

            return train_dataset, val_dataset
        else: 
            test_dataset = BloodMNIST(split='test', download=True, size=224)
            return test_dataset