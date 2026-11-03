"""Validation at the dataset and model boundaries."""

import pandas as pd
import pytest
import torch

from ethnicolr2.dataset import EthniDataset
from ethnicolr2.models import LSTM


def test_dataset_requires_names_and_positive_sequence_length():
    with pytest.raises(ValueError, match="__name"):
        EthniDataset(pd.DataFrame({"other": ["abc"]}), "abc", 4, 3)
    with pytest.raises(ValueError, match="max_name must be positive"):
        EthniDataset(pd.DataFrame({"__name": ["abc"]}), "abc", 0, 3)


def test_dataset_accepts_tensor_index_and_returns_empty_untransformed_tensor():
    dataset = EthniDataset(pd.DataFrame({"__name": ["abc"]}), "abc", 4, 3)
    name, encoded = dataset[torch.tensor(0)]
    assert name == "abc"
    assert encoded.numel() == 0
    for index in [-1, 1]:
        with pytest.raises(IndexError, match="out of range"):
            dataset[index]


def test_dataset_rejects_nonstring_values():
    dataset = EthniDataset(pd.DataFrame({"__name": [123]}), "abc", 4, 3)
    with pytest.raises(ValueError, match="Expected string name"):
        dataset[0]


@pytest.mark.parametrize(
    "dimension", ["input_size", "hidden_size", "output_size", "num_layers"]
)
def test_lstm_rejects_nonpositive_dimensions(dimension):
    dimensions = {"input_size": 5, "hidden_size": 3, "output_size": 2, "num_layers": 1}
    dimensions[dimension] = 0
    with pytest.raises(ValueError, match=f"{dimension} must be positive"):
        LSTM(**dimensions)


def test_lstm_rejects_invalid_input_type_and_shape():
    model = LSTM(5, 3, 2)
    with pytest.raises(TypeError, match="Expected torch.Tensor"):
        model([1, 2])
    with pytest.raises(ValueError, match="Expected 2D tensor"):
        model(torch.tensor([1, 2]))
