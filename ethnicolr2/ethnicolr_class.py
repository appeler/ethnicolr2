#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import joblib

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from pkg_resources import resource_filename
from itertools import chain

from tqdm import tqdm
tqdm.pandas()

import torch
from torch.utils.data import DataLoader

from .dataset import EthniDataset
from .models import LSTM

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

        df.dropna(subset=[col], inplace = True)
        if df.shape[0] == 0:
            raise Exception("The name column has no non-NaN values.")

        df.drop_duplicates(subset = [col], inplace = True)

        return df

    @staticmethod
    def n_grams(seq, n:int=1):
        """Returns an iterator over the n-grams given a listTokens"""
        shiftToken = lambda i: (el for j, el in enumerate(seq) if j >= i)
        shiftedTokens = (shiftToken(i) for i in range(n))
        tupleNGrams = zip(*shiftedTokens)
        return tupleNGrams

    @staticmethod
    def range_ngrams(listTokens, ngramRange=(1, 2)):
        """Returns an iterator over all n-grams for n in range(ngramRange)
          given a listTokens.
        """
        ngrams = (ngramRange[0], ngramRange[1] + 1)
        return chain(*(EthnicolrModelClass.n_grams(listTokens, i) for i in range(*ngramRange)))


    @staticmethod
    def find_ngrams(vocab, text: str, n) -> list:
        """Find and return list of the index of n-grams in the vocabulary list.

        Generate the n-grams of the specific text, find them in the vocabulary list
        and return the list of index have been found.

        Args:
            vocab (:obj:`list`): Vocabulary list.
            text (str): Input text
            n (int or tuple): N-grams or tuple of range N-grams

        Returns:
            list: List of the index of n-grams in the vocabulary list.

        """

        wi = []

        if type(n) is tuple:
            a = EthnicolrModelClass.range_ngrams(text, n)
        else:
            a = zip(*[text[i:] for i in range(n)])

        for i in a:
            w = "".join(i)
            try:
                idx = vocab.index(w)
            except Exception as e:
                idx = 0
            wi.append(idx)
        return wi


    @classmethod
    def transform_and_pred(cls,
        df: pd.DataFrame, newnamecol: str, vocab_fn: str , race_fn: str, model_fn: str, ngrams, maxlen: int, num_iter: int, conf_int: float
    ) -> pd.DataFrame:

        VOCAB = resource_filename(__name__, vocab_fn)
        MODEL = resource_filename(__name__, model_fn)
        RACE = resource_filename(__name__, race_fn)

        df = EthnicolrModelClass.test_and_norm_df(df, newnamecol)

        df[newnamecol] = df[newnamecol].str.strip().str.title()
        df["rowindex"] = df.index

        if cls.model is None:
            vdf = pd.read_csv(VOCAB)
            cls.vocab = vdf.vocab.tolist()

            rdf = pd.read_csv(RACE)
            cls.race = rdf.race.tolist()

            cls.model = load_model(MODEL)

        # build X from index of n-gram sequence
        X = np.array(df[newnamecol].apply(lambda c: EthnicolrModelClass.find_ngrams(cls.vocab, c, ngrams)))
        X = sequence.pad_sequences(X, maxlen=maxlen)

        if conf_int == 1:
            # Predict
            proba = cls.model(X, training=False).numpy()
            pdf = pd.DataFrame(proba, columns=cls.race)
            pdf["__race"] = np.argmax(proba, axis=-1)
            pdf["race"] = pdf["__race"].apply(lambda c: cls.race[int(c)])
            del pdf["__race"]
            final_df = pd.concat([df.reset_index(drop=True),
                                  pdf.reset_index(drop=True)], axis=1 )
        else:
            # define the quantile ranges for the confidence interval
            lower_perc = (0.5 - (conf_int / 2)) * 100
            upper_perc = (0.5 + (conf_int / 2)) * 100

            # Predict
            pdf = pd.DataFrame()

            for _ in range(num_iter):
                pdf = pd.concat([pdf, pd.DataFrame(cls.model(X, training=True))])
            print(cls.race)
            pdf.columns = cls.race
            pdf["rowindex"] = pdf.index

            res = (
                pdf.groupby("rowindex")
                .agg(
                    [
                        np.mean,
                        np.std,
                        lambda x: np.percentile(x, q=lower_perc),
                        lambda x: np.percentile(x, q=upper_perc),
                    ]
                )
                .reset_index()
            )
            res.columns = [f"{i}_{j}" for i, j in res.columns]
            res.columns = res.columns.str.replace("<lambda_0>", "lb")
            res.columns = res.columns.str.replace("<lambda_1>", "ub")
            res.columns = res.columns.str.replace("rowindex_", "rowindex")

            means = list(filter(lambda x: "_mean" in x, res.columns))
            res["race"] = res[means].idxmax(axis=1).str.replace("_mean", "")

            for suffix in ["_lb", "ub"]:
                conv_filt = list(filter(lambda x: suffix in x, res.columns))
                res[conv_filt] = res[conv_filt].to_numpy().astype(float)

            final_df = df.merge(res, on="rowindex", how="left")

        del final_df['rowindex']
        del df['rowindex']

        return final_df


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
    def predict(cls, df: pd.DataFrame, vocab_fn:str,  model_fn: str) -> pd.DataFrame:
        """
        predict based on the model and vocab
        """
        MODEL = resource_filename(__name__, model_fn)
        VOCAB = resource_filename(__name__, vocab_fn)

        vectorizer = joblib.load(VOCAB)
        all_categories = ['asian', 'hispanic', 'nh_black', 'nh_white', 'other']
        if 'fullname' in model_fn:
            max_name = 47
        elif 'census' in model_fn:
            max_name = 15
            all_categories = ['nh_white', 'nh_black', 'hispanic', 'asian', 'other']
        else:
            max_name = 30
        n_categories = len(all_categories)
        vocab = list(vectorizer.get_feature_names_out())
        all_letters = ''.join(vocab)
        n_letters = len(vocab)
        oob = n_letters + 1
        vocab_size = oob + 1
        batch_size = 64
        n_hidden = 256

        dataset = EthniDataset(df, all_letters, max_name, oob, transform=EthnicolrModelClass.lineToTensor) #noqa
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        torch.manual_seed(42)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = LSTM(vocab_size, n_hidden, n_categories, num_layers=2)
        model.load_state_dict(torch.load(MODEL, map_location=device))
        model.to(device)

        # Set the model to evaluation mode
        model.eval()

        # List to hold the predictions
        predictions = []
        names = []
        # Disable gradient calculations
        with torch.no_grad():
            # Loop over the batches
            for batch in tqdm(dataloader):
                # Move the batch to the device the model is on
                nms = list(batch[0])
                tns = batch[1].to(device)
                # Compute the predictions
                outputs = model(tns)
                outputs = torch.argmax(outputs, dim=1)
                # Move the predictions to the CPU and convert to numpy arrays
                outputs = outputs.cpu().numpy()
                # Append the predictions to the list
                predictions.extend(list(outputs))
                names.extend(nms)

        results_df = pd.DataFrame({'names': names, 'predictions': predictions})
        results_df['preds'] = results_df['predictions'].apply(lambda x: all_categories[x])
        results_df = results_df.drop(columns=['predictions'])

        final_df = pd.merge(df, results_df, left_on=['__name'], right_on=['names'], how='left')
        final_df = final_df.drop(columns=['names'])
        return final_df
