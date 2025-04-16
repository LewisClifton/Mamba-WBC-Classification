import pandas as pd
import numpy as np

from torch.utils.data import Dataset
import torch

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform, test=False):
        self.dataset = dataset
        self.transform = transform
        self.test=test

    def __getitem__(self, idx):

        image, label, image_name = self.dataset[idx]
        image = self.transform(image)
        
        return image, label, image_name

    def __len__(self):
        return len(self.dataset)
    

class EnsembleDataset(Dataset):
    def __init__(self, dataset, base_model_transforms, test=False):
        """
        Args:
            image_paths (list): List of image file paths.
            transforms_list (list): List of transformations (one per model).
        """
        self.dataset = dataset
        self.base_model_transforms = base_model_transforms
        self.test = test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        images, label, image_name = self.dataset[idx]

        transformed_images = [transform(images) for transform in self.base_model_transforms]
        return transformed_images, label, image_name
    

class MetaLearnerDataset(Dataset):
    def __init__(self, model_outputs_paths, dataset):
        """
        Args:
            image_paths (list): List of image file paths.
            transforms_list (list): List of transformations (one per model).
        """

         # image dataset
        self.dataset = dataset # (image, true_label, image_name)

        all_model_outputs_df = None

        # Combine all the models outputs
        for model_outputs_path in model_outputs_paths:
            
            # Get this model outputs 
            model_outputs_df = pd.read_csv(model_outputs_path['trained_model_path']) # (image_name, model_x_out_1, model_x_out_2, ... , model_x_out_8)

            if all_model_outputs_df is None:
                all_model_outputs_df = model_outputs_df
            else:
                # Merge dataset with model outputs using the image name
                all_model_outputs_df = pd.merge(all_model_outputs_df, model_outputs_df, on="name", how="inner")

        self.all_model_outputs_df = all_model_outputs_df # (image_name, model_x_out_1, model_x_out_2, ... , model_x_out_8, model_y_out_1, model_y_out_2, ... , model_y_out_8)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _, label, image_name = self.dataset[idx]

        # Get all the model outputs
        model_outputs = self.all_model_outputs_df[self.all_model_outputs_df['name'] == image_name].drop(columns=['name']).to_numpy().T

        model_outputs = torch.tensor(model_outputs, dtype=torch.float).squeeze(-1)

        return model_outputs, label, image_name 
    

def get_dataset(dataset_config, dataset_download_dir, test=False):
    """
    Initialise fresh model prior to training

    Args:
        model_type (string): Model type e.g. 'swin'
        num_classes (int): Number of WBC classification classes

    Returns:
        torch.nn.Module: Initialised model ready for training
        dict: Dictionary containing the required data transforms. Use "train"/"val" keys to access training/validation data transforms
    """

    dataset_name = dataset_config['name']

    # Get the required dataset
    if dataset_name == 'chula':
        from .chula import get

        dataset = get(dataset_config, test)

    elif dataset_name == 'bloodmnist':
        from .bloodmnist import get

        # BloodMNIST has predefined train/val/test splits
        if test:
            test_dataset = get(dataset_config, dataset_download_dir, test)
            return test_dataset
        else:
            train_dataset, val_dataset = get(dataset_config, dataset_download_dir, test)
            return train_dataset, val_dataset 

    elif dataset_name == 'foo':
        pass

    if 'base_model_outputs_paths' in dataset_config.keys():

        dataset = MetaLearnerDataset(dataset_config['base_model_outputs_paths'], dataset)

    return dataset