from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import pandas as pd

from typing import Tuple

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    
    def __init__(self, data: pd.DataFrame, mode: str) ->None:
        self.data = data
        self.mode = mode
        if mode == 'train':
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.RandomVerticalFlip(),
                tv.transforms.RandomRotation(30),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        else:
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        filename = self.data.iloc[index, 0]
        crack = self.data.iloc[index, 1]
        inactive = self.data.iloc[index, 2]
        image_path = Path(filename)
        image = imread(image_path)
        image = gray2rgb(image)
        image = self.transform(image)
        label = torch.tensor([crack, inactive])
        return image, label

    
    
        
