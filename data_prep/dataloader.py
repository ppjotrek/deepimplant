import json
import numpy as np
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset


BATCH_SIZE = 2

class MeshDataset(Dataset):
    def __init__(self, targ_dir: str, transform=None, shuffle=True):
        self.paths = list(Path(targ_dir).glob("*.json"))
        self.transform = transform
        self.shuffle = shuffle
        #TODO: shuffle paths
        #TODO: implement transform
        #TODO: implement getting data from json file -> here or in getitem?
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx) -> tuple(torch.Tensor, int):
        pass
        #return tensor + label, TODO: which tensor?