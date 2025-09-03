#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import joblib
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

try:
    from importlib.resources import files
except ImportError:
    # Fallback for Python < 3.9
    from pkg_resources import resource_filename
from itertools import chain

from tqdm import tqdm

tqdm.pandas()

import torch
from torch.utils.data import DataLoader

from .dataset import EthniDataset
from .models import LSTM

# Model parameter constants
MAX_NAME_FULLNAME = 47
MAX_NAME_FLORIDA = 30
MAX_NAME_CENSUS = 15
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS = 2


class EthnicolrModelClass:
    vocab: Optional[List[str]] = None
    race: Optional[List[str]] = None
    model: Optional[torch.nn.Module] = None
    model_year: Optional[int] = None

    @staticmethod
    def test_and_norm_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Validates and normalizes DataFrame for prediction.
        
        Args:
            df: Input DataFrame
            col: Column name to validate and process
            
        Returns:
            Cleaned DataFrame with duplicates and NaN values removed
            
        Raises:
            ValueError: If column doesn't exist or contains no valid data
        """
        if col and (col not in df.columns):
            raise ValueError(f"Column '{col}' not found in DataFrame. Available columns: {list(df.columns)}")

        df.dropna(subset=[col], inplace=True)
        if df.shape[0] == 0:
            raise ValueError(f"Column '{col}' contains no non-NaN values.")

        df.drop_duplicates(subset=[col], inplace=True)

        return df

    @staticmethod
    def lineToTensor(line: str, all_letters: str, max_name: int, oob: int) -> torch.Tensor:
        """Convert a name string to a tensor of character indices.
        
        Args:
            line: Input name string
            all_letters: String containing all valid characters
            max_name: Maximum name length (longer names are truncated)
            oob: Out-of-bounds index for unknown characters
            
        Returns:
            Tensor of character indices with shape (max_name,)
        """
        if not isinstance(line, str):
            raise TypeError(f"Expected string input, got {type(line)}")
        if max_name <= 0:
            raise ValueError(f"max_name must be positive, got {max_name}")
            
        # Truncate if name is longer than max_name
        if len(line) > max_name:
            line = line[:max_name]
            
        tensor = torch.ones(max_name, dtype=torch.long) * oob
        for li, letter in enumerate(line):
            char_idx = all_letters.find(letter)
            tensor[li] = char_idx if char_idx != -1 else oob
        return tensor

    @classmethod
    def predict(cls, df: pd.DataFrame, vocab_fn: str, model_fn: str) -> pd.DataFrame:
        """Generate race/ethnicity predictions for names in DataFrame.
        
        Args:
            df: DataFrame containing name data with '__name' column
            vocab_fn: Path to vocabulary file (.joblib)
            model_fn: Path to trained model file (.pt)
            
        Returns:
            DataFrame with original data plus 'preds' and 'probs' columns
            
        Raises:
            FileNotFoundError: If model or vocabulary files don't exist
            ValueError: If DataFrame is empty or malformed
            RuntimeError: If model loading or prediction fails
        """
        try:
            # Use modern importlib.resources for Python >= 3.9
            import ethnicolr2

            MODEL = str(files(ethnicolr2).joinpath(model_fn))
            VOCAB = str(files(ethnicolr2).joinpath(vocab_fn))
        except (NameError, AttributeError):
            # Fallback to pkg_resources for older Python versions
            MODEL = resource_filename(__name__, model_fn)
            VOCAB = resource_filename(__name__, vocab_fn)

        vectorizer = joblib.load(VOCAB)
        all_categories = ["asian", "hispanic", "nh_black", "nh_white", "other"]
        if "fullname" in model_fn:
            max_name = MAX_NAME_FULLNAME
        elif "census" in model_fn:
            max_name = MAX_NAME_CENSUS
            all_categories = ["nh_white", "nh_black", "hispanic", "asian", "other"]
        else:
            max_name = MAX_NAME_FLORIDA
        n_categories = len(all_categories)
        vocab = list(vectorizer.get_feature_names_out())
        all_letters = "".join(vocab)
        n_letters = len(vocab)
        oob = n_letters + 1
        vocab_size = oob + 1
        batch_size = BATCH_SIZE
        n_hidden = HIDDEN_SIZE

        dataset = EthniDataset(
            df, all_letters, max_name, oob, transform=EthnicolrModelClass.lineToTensor
        )  # noqa
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        torch.manual_seed(42)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = LSTM(vocab_size, n_hidden, n_categories, num_layers=NUM_LAYERS)
        model.load_state_dict(torch.load(MODEL, map_location=device))
        model.to(device)

        # Set the model to evaluation mode
        model.eval()

        # List to hold the predictions
        predictions = []
        names = []
        softprobs = []
        # Disable gradient calculations
        with torch.no_grad():
            # Loop over the batches
            for batch in tqdm(dataloader):
                # Move the batch to the device the model is on
                nms = list(batch[0])
                tns = batch[1].to(device)
                # Compute the predictions
                outputs = model(tns)
                # get soft probabilities
                probs = torch.softmax(outputs, dim=1)
                # match with all_categories and store probs as json in softprobs
                softprobs.extend(
                    [dict(zip(all_categories, p)) for p in probs.cpu().numpy()]
                )
                outputs = torch.argmax(outputs, dim=1)
                # Move the predictions to the CPU and convert to numpy arrays
                outputs = outputs.cpu().numpy()
                # Append the predictions to the list
                predictions.extend(list(outputs))
                names.extend(nms)

        results_df = pd.DataFrame({"names": names, "predictions": predictions})
        results_df["preds"] = results_df["predictions"].apply(
            lambda x: all_categories[x]
        )
        results_df["probs"] = softprobs
        results_df = results_df.drop(columns=["predictions"])

        final_df = pd.merge(
            df, results_df, left_on=["__name"], right_on=["names"], how="left"
        )
        final_df = final_df.drop(columns=["names"])
        return final_df
