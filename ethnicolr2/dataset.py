from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
import pandas as pd


class EthniDataset(Dataset):
    """PyTorch Dataset for ethnicolr name data.
    
    Args:
        data_df: DataFrame containing name data with '__name' column
        all_letters: String of all valid characters in vocabulary
        max_name: Maximum length for name sequences
        oob: Out-of-bounds index for unknown characters
        transform: Optional transformation function for names
    """

    def __init__(
        self, 
        data_df: pd.DataFrame, 
        all_letters: str, 
        max_name: int, 
        oob: int, 
        transform: Optional[Callable[[str, str, int, int], torch.Tensor]] = None
    ):
        if not isinstance(data_df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(data_df)}")
        if '__name' not in data_df.columns:
            raise ValueError("DataFrame must contain '__name' column")
        if not isinstance(all_letters, str):
            raise TypeError(f"Expected string for all_letters, got {type(all_letters)}")
        if max_name <= 0:
            raise ValueError(f"max_name must be positive, got {max_name}")
            
        self.df = data_df
        self.transform = transform
        self.all_letters = all_letters
        self.max_name = max_name
        self.oob = oob

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (name_string, name_tensor)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if not isinstance(idx, int) or idx < 0 or idx >= len(self.df):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.df)}")
            
        name = self.df.iloc[idx, self.df.columns.get_loc("__name")]
        if not isinstance(name, str):
            raise ValueError(f"Expected string name, got {type(name)}: {name}")
            
        if self.transform:
            name_ids = self.transform(name, self.all_letters, self.max_name, self.oob)
        else:
            name_ids = torch.tensor([])  # Empty tensor if no transform
        return name, name_ids
