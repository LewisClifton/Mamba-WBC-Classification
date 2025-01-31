import os

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch

from medmnist import BloodMNIST

class WBC5000dataset(Dataset):
    def __init__(self, images_path, labels_path, wbc_types=['BNE', 'SNE', 'Basophil', 'Eosinophil', 'Monocyte', 'Lymphocyte']):
        super(WBC5000dataset, self).__init__()

        self.images_path = images_path

        self.labels = WBC5000dataset._get_labels(labels_path, wbc_types)

        self.wbc_types = wbc_types

        # wbc type is semantic, class is numerical classes used by the model
        self.wbc_type_to_class = {label: idx for idx, label in enumerate(wbc_types)}
        self.class_to_wbc_type = {idx: label for idx, label in enumerate(wbc_types)}
    
    @staticmethod
    def _get_labels(labels_path, wbc_types):
        # Get cleaned labels 

        # Read labels
        labels = pd.read_csv(labels_path)

        # Remove and rename columns
        labels = labels[['Order.jpg', 'Summary by 5 experts']].rename(columns={'Order.jpg': 'name', 'Summary by 5 experts': 'label'})
        
        # Fix typos in labels
        corrected_labels = {
            'Eosinophill': 'Eosinophil',
            'Monocyte ': 'Monocyte',
            'SNE\t': 'SNE',
            'Myeolblast': 'Myeloblast',
            'Atypical lymphocyte': 'Atypical Lymphocyte', 
            'Smudge cell': 'Smudge Cell',
            'Giant platelet': 'Giant Platelet'
        }
        labels['label'] = labels['label'].replace(corrected_labels).str.strip()

        # Only keep required wbc types
        labels = labels[labels['label'].isin(wbc_types)]

        return labels

    def get_wbc_type_from_class(self, int):
        return self.class_to_wbc_type[int]

    def get_class_from_wbc_type(self, label):
        return self.wbc_type_to_class[label]

    def __getitem__(self, idx):
        
        image_path = os.path.join(self.images_path, self.labels.iloc[idx]['name'])
        image = Image.open(image_path)

        label = self.labels.iloc[idx]['label']
        label = self.get_class_from_wbc_type(label)
        
        return image, label
    
    def __len__(self):
        return len(self.labels)

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)