import torch
from torch.utils.data import Dataset


class EthniDataset(Dataset):
    """Ethnicolr Dataset
    """
    def __init__(self, data_df, all_letters, max_name, oob, transform=None):
        self.df = data_df
        self.transform = transform
        self.all_letters = all_letters
        self.max_name = max_name
        self.oob = oob


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = self.df.iloc[idx, self.df.columns.get_loc('__name')]
        if self.transform:
            name_ids = self.transform(name, self.all_letters, self.max_name, self.oob)
        return name, name_ids
    