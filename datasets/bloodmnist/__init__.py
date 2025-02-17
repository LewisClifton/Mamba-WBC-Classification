import os

from medmnist import BloodMNIST

def get(test=False):

    if not test:
        train_dataset = BloodMNIST(split='train', download=True, size=224, root=os.path.join(os.getcwd(), 'datasets/bloodmnist/data/'))
        val_dataset = BloodMNIST(split='val', download=True, size=224, root=os.path.join(os.getcwd(), 'datasets/bloodmnist/data/'))
        return train_dataset, val_dataset
    else: 
        test_dataset = BloodMNIST(split='test', download=True, size=224, root=os.path.join(os.getcwd(), 'datasets/bloodmnist/data/'))
        return test_dataset

    