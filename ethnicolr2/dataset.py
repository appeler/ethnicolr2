from collections.abc import Callable

import pandas as pd
import torch
from torch.utils.data import Dataset


class EthniDataset(Dataset[tuple[str, torch.Tensor]]):
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
        transform: Callable[[str, str, int, int], torch.Tensor] | None = None,
    ):
        if "__name" not in data_df.columns:
            raise ValueError("DataFrame must contain '__name' column")
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

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (name_string, name_tensor)
        """
        if torch.is_tensor(idx):
            idx_val = int(idx.item())  # Convert tensor to int safely
        else:
            idx_val = idx

        if idx_val < 0 or idx_val >= len(self.df):
            raise IndexError(
                f"Index {idx_val} out of range for dataset of size {len(self.df)}"
            )

        name = self.df.iloc[idx_val, self.df.columns.get_loc("__name")]
        if not isinstance(name, str):
            raise ValueError(f"Expected string name, got {type(name)}: {name}")

        if self.transform:
            name_ids = self.transform(name, self.all_letters, self.max_name, self.oob)
        else:
            name_ids = torch.tensor([])  # Empty tensor if no transform
        return name, name_ids
