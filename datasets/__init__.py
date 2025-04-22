import pandas as pd
import numpy as np

from torch.utils.data import Dataset
import torch
from torchvision import transforms

# Dataset wrappaer that applies model transforms to images before they are used for model input
class TransformedDataset(Dataset):
    def __init__(self, dataset, transform = None):
        self.dataset = dataset

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, idx):

        image, label, image_name = self.dataset[idx]
        image = self.transform(image)
        
        return image, label, image_name

    def __len__(self):
        return len(self.dataset)
    

# Dataset for ensemble which returns image batches using the transform for each base model
class EnsembleDataset(Dataset):
    def __init__(self, dataset, base_model_transforms):
        """
        Args:
            image_paths (list): List of image file paths.
            transforms_list (list): List of transformations (one per model).
        """
        self.dataset = dataset
        self.base_model_transforms = base_model_transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        images, label, image_name = self.dataset[idx]

        transformed_images = [transform(images) for transform in self.base_model_transforms]
        return transformed_images, label, image_name
    

# Dataset for the stacking meta learner which comprises the out of fold outputs for each base model
class MetaLearnerDataset(Dataset):
    def __init__(self, model_outputs_paths, dataset):
        """
        Args:
            image_paths (list): List of image file paths.
            transforms_list (list): List of transformations (one per model).
        """

         # Image dataset
        self.dataset = dataset # (image, true_label, image_name)

        all_model_outputs_df = None

        # Combine all the models outputs
        for model_outputs_path in model_outputs_paths:
            
            # Get this model outputs 
            model_outputs_df = pd.read_csv(model_outputs_path['path']) # (image_name, model_x_out_1, model_x_out_2, ... , model_x_out_8)

            if all_model_outputs_df is None:
                all_model_outputs_df = model_outputs_df
            else:
                # Merge dataset with model outputs using the image name
                all_model_outputs_df = pd.merge(all_model_outputs_df, model_outputs_df, on="name", how="inner")

        self.all_model_outputs_df = all_model_outputs_df # (image_name, model_x_out_1, model_x_out_2, ... , model_x_out_8, model_y_out_1, model_y_out_2, ... , model_y_out_8)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the image name and label - we don't need the image itself as the meta learner is just trained on base model outputs
        _, label, image_name = self.dataset[idx]

        # Get all the model outputs
        model_outputs = self.all_model_outputs_df[self.all_model_outputs_df['name'] == image_name].drop(columns=['name']).to_numpy().T

        # Convert to tensor
        model_outputs = torch.tensor(model_outputs, dtype=torch.float).squeeze(-1)

        return model_outputs, label, image_name 
    

def get_dataset(dataset_config, dataset_download_dir, test=False):
    """
    Get dataset used for training / evaluation.

    Args:
        dataset_config (dict): Dataset configuration file
        dataset_download_dir (string): Directory to download the dataset to if necessary (e.g. for BloodMNIST)
        test (bool): Whether to return the train split or test split

    Returns:
        torch.utils.data.Dataset: Loaded dataset
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