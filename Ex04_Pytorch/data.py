from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import torchvision.transforms as transforms
import pandas as pd

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data: pd.DataFrame, mode: str):
        self.data_frame = data
        self.transform = None
        # data augmentation
        if mode == 'val':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(train_mean, train_std)

            ])

        elif mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),

                # Dynamic Data Augmentation
                #transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomVerticalFlip(p=0.5),
                #transforms.RandomRotation(degrees=(-90, 90)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(train_mean, train_std)
            ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = imread('./' + self.data_frame.iloc[idx, 0])
        image = gray2rgb(image)
        if self.transform:
            image = self.transform(image)
        data = self.data_frame.iloc[idx, 1:]
        data = torch.tensor(data, dtype=torch.float32)
        sample = (image, data)
        return sample
