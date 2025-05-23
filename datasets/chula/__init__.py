import os
from PIL import Image
import pandas as pd
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch


class ChulaWBC5000(Dataset):
    def __init__(self, images_path, labels_path, wbc_types=['BNE', 'SNE', 'Basophil', 'Eosinophil', 'Monocyte', 'Lymphocyte'], test=False):
        super(ChulaWBC5000, self).__init__()

        self.images_path = images_path

        labels = pd.read_csv(labels_path)
        self.labels = labels[labels['label'].isin(wbc_types)]
        self.labels_path = labels_path

        self.wbc_types = wbc_types

        # wbc type is semantic, class is numerical classes used by the model
        self.wbc_type_to_class = {label: idx for idx, label in enumerate(wbc_types)}
        self.class_to_wbc_type = {idx: label for idx, label in enumerate(wbc_types)}

        self.test = test

    def get_wbc_type_from_class(self, class_):
        '''
        Convert integer class value to corresponding WBC type

        Args:
            class_(int): Integer class value to be converted

        Return:
            string: WBC type label
        '''
        return self.class_to_wbc_type[class_]

    def get_class_from_wbc_type(self, label):
        '''
        Convert WBC type to corresponding integer class value 

        Args:
            label: WBC type label
            
        Return:
            int: Integer class value to be converted
        '''
        return self.wbc_type_to_class[label]

    def __getitem__(self, idx):

        # Get image
        image_name = self.labels.iloc[idx]['name']
        image_path = os.path.join(self.images_path, image_name)
        image = Image.open(image_path)

        # Get label and convert to numerical
        label = self.labels.iloc[idx]['label']
        label = self.get_class_from_wbc_type(label)

        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        
        return image, label, image_name
    
    def __len__(self):
        return len(self.labels)


def get(dataset_config, test=False):

    # Get train or test dataset
    if test:
        labels_path = dataset_config['test_labels_path']
    else:
        labels_path = dataset_config['train_labels_path']

    dataset = ChulaWBC5000(dataset_config['images_dir'], labels_path, wbc_types=dataset_config['classes'], test=test)

    return dataset