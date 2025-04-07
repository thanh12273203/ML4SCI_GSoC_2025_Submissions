# Import necessary dependencies
from typing import Optional, Callable, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
    
# Dataset for pretraining the masked transformer autoencoder
class JetsDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, transform: Optional[Callable] = None):
        self.X = X.to_numpy().astype(np.float32)
        self.y = y.to_numpy().astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        sample = self.X[idx]  # (21,)
        label = self.y[idx]

        if self.transform:
            jets = self.transform(jets)

        return torch.tensor(sample).unsqueeze(-1), torch.tensor(label)
    
# Dataset for self-supervised learning following the VICReg approach
class VICRegUnlabelledDataset(Dataset):
    def __init__(self, X: np.ndarray, transform: Optional[Callable] = None):
        self.X = torch.from_numpy(X.copy()).float()
        self.transform = transform

    def __len__(self) -> int:
        return self.X.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = self.X[idx]

        if self.transform is not None:
            # Apply the same transform function twice to get two views of the same data
            x1 = self.transform(x)
            x2 = self.transform(x)

        return x1, x2
    
# Dataset for self-supervised learning following the VICReg approach with lazy loading from HDF5
class VICRegLazyLoadingDataset(Dataset):
    def __init__(self, h5_path: str, transform: Optional[Callable] = None):
        self.h5_path = h5_path
        self.transform = transform
        
        # Open the file once to get the dataset length
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['jet'].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        if not hasattr(self, '_h5_file'):
            self._h5_file = h5py.File(self.h5_path, 'r')

        x = self._h5_file['jet'][idx]
        x = torch.from_numpy(x).float()
        
        if self.transform is not None:
            # Apply the same transform function twice to get two views of the same data
            x1 = self.transform(x)
            x2 = self.transform(x)
        else:
            x1 = x
            x2 = x

        return x1, x2