import os

from medmnist import BloodMNIST

def get(test=False):

    # Set up path for the dataset 
    path = os.path.join(os.getcwd(), 'datasets/bloodmnist/data/')
    if not os.path.isdir(path):
        os.mkdir(path)

    # Get the dataset
    if not test:
        train_dataset = BloodMNIST(split='train', download=True, size=224)
        val_dataset = BloodMNIST(split='val', download=True, size=224)
        return train_dataset, val_dataset
    else: 
        test_dataset = BloodMNIST(split='test', download=True, size=224)
        return test_dataset

    