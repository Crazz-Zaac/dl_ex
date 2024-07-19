from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

from typing import Tuple

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    
    def __init__(self, data: pd.DataFrame, mode: str) ->None:
        self.data = data
        self.mode = mode
        self.transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.Resize((128, 128)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=train_mean, std=train_std)
        ])
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.data.iloc[idx, 0]
        img = imread(img_path)
        if len(img.shape) == 2:
            img = gray2rgb(img)
        img = self.transform(img)
        label = self.data.iloc[idx, 1]
        return img, label

    
    
        
