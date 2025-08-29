#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import joblib

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
    vocab = None
    race = None
    model = None
    model_year = None

    @staticmethod
    def test_and_norm_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Handles cases like:
        - column doesn't exist, nukes missing rows

        """
        if col and (col not in df.columns):
            raise Exception(f"The column {col} doesn't exist in the dataframe.")

        df.dropna(subset=[col], inplace=True)
        if df.shape[0] == 0:
            raise Exception("The name column has no non-NaN values.")

        df.drop_duplicates(subset=[col], inplace=True)

        return df

    @staticmethod
    def lineToTensor(line, all_letters, max_name, oob):
        """
        Turn a line into a <max_name x 1 x n_letters>,
        or an array of one-hot letter vectors
        """
        # if name is more than max_name
        if len(line) > max_name:
            line = line[:max_name]
        tensor = torch.ones(max_name) * oob
        for li, letter in enumerate(line):
            tensor[li] = all_letters.find(letter)
        return tensor

    @classmethod
    def predict(cls, df: pd.DataFrame, vocab_fn: str, model_fn: str) -> pd.DataFrame:
        """
        predict based on the model and vocab
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
