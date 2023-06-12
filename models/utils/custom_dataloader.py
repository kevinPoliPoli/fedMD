import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.X = data_dict["X"]
        self.y = data_dict["y"]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        return image, label



class ConsensusDataset(Dataset):
    def __init__(self, dataset, consensus):
        self.X = dataset["X"]
        self.y = consensus
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        return image, label