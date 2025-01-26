from torch.utils.data import Dataset
import os
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd


class WBC5000dataset(Dataset):
    def __init__(self, images_path, labels_path):
        super(WBC5000dataset, self).__init__()

        self.images_path = images_path

        labels = pd.read_csv(labels_path)
        self.labels = labels[['Order.jpg', 'Summary by 5 experts']].rename(columns={'Order.jpg': 'name', 'Summary by 5 experts': 'label'})

    def __getitem__(self, idx):
        
        image_path = os.path.join(self.images_path, self.labels.iloc[idx]['name'])
        image = Image.open(image_path)
        image = np.array(image)
        image = transforms.ToTensor()(image)

        label = self.labels.iloc[idx]['label']

        return image, label
    
    def __len__(self):
        return len(self.labels)