import os

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch

class ChulaWBC5000(Dataset):
    def __init__(self, images_path, labels_path, wbc_types=['BNE', 'SNE', 'Basophil', 'Eosinophil', 'Monocyte', 'Lymphocyte']):
        super(ChulaWBC5000, self).__init__()

        self.images_path = images_path

        self.labels = ChulaWBC5000._get_labels(labels_path, wbc_types)

        self.wbc_types = wbc_types

        # wbc type is semantic, class is numerical classes used by the model
        self.wbc_type_to_class = {label: idx for idx, label in enumerate(wbc_types)}
        self.class_to_wbc_type = {idx: label for idx, label in enumerate(wbc_types)}
    
    @staticmethod
    def _get_labels(labels_path, wbc_types):
        """
        Get clean labels from the Chula dataset

        Args:
            labels_path(string): Path of labels.csv
            wbc_types(list[string]): List of WBC type labels to be considered (ignore rest)

        Returns:
            pandas.Dataframe: Dataframe containing image names and cleaned and filtered WBC labels
        """

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
        
        image_path = os.path.join(self.images_path, self.labels.iloc[idx]['name'])
        image = Image.open(image_path)

        label = self.labels.iloc[idx]['label']
        label = self.get_class_from_wbc_type(label)

        label = torch.tensor(label, dtype=torch.long).unsqueeze(0)
        
        return image, label
    
    def __len__(self):
        return len(self.labels)
    

def get(dataset_config, test=False):
    dataset = ChulaWBC5000(dataset_config['images_dir'], dataset_config['labels_path'], wbc_types=dataset_config['classes'])

    # TO DO: CHULA TRAIN TEST SPLIT

    return dataset