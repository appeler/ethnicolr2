#!/usr/bin/env python

import os
import time
from threading import Lock
from typing import Any

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from importlib.resources import files
except ImportError:
    # Fallback for Python < 3.9
    from pkg_resources import resource_filename

from tqdm import tqdm

from .dataset import EthniDataset
from .models import LSTM

tqdm.pandas()

# Module-level caching infrastructure
_MODEL_CACHE: dict[str, tuple[torch.nn.Module, Any, dict[str, Any]]] = {}
_CACHE_LOCK = Lock()
_CACHE_STATS = {"hits": 0, "misses": 0, "loads": 0}

# Cache configuration
CACHE_ENABLED = os.getenv("ETHNICOLR_CACHE_ENABLED", "true").lower() == "true"
MAX_CACHED_MODELS = int(os.getenv("ETHNICOLR_MAX_CACHED_MODELS", "3"))


def _get_cache_key(model_fn: str, vocab_fn: str) -> str:
    """Generate unique cache key for model files."""
    try:
        # Include file modification time to handle model updates
        model_mtime = os.path.getmtime(model_fn) if os.path.exists(model_fn) else 0
        vocab_mtime = os.path.getmtime(vocab_fn) if os.path.exists(vocab_fn) else 0
        return f"{model_fn}#{vocab_fn}#{model_mtime}#{vocab_mtime}"
    except OSError:
        # Fallback for resource files
        return f"{model_fn}#{vocab_fn}"


def _load_and_cache_model(
    model_fn: str, vocab_fn: str
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Load model and cache with computed metadata."""

    # Load vectorizer
    vectorizer = joblib.load(vocab_fn)

    # Determine model type and configuration
    all_categories = ["asian", "hispanic", "nh_black", "nh_white", "other"]
    if "fullname" in model_fn:
        max_name = MAX_NAME_FULLNAME
    elif "census" in model_fn:
        max_name = MAX_NAME_CENSUS
        all_categories = ["nh_white", "nh_black", "hispanic", "asian", "other"]
    else:
        max_name = MAX_NAME_FLORIDA

    # Pre-compute expensive operations
    vocab = list(vectorizer.get_feature_names_out())
    all_letters = "".join(vocab)
    n_letters = len(vocab)
    oob = n_letters + 1
    vocab_size = oob + 1
    n_categories = len(all_categories)

    # Load and configure model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LSTM(vocab_size, HIDDEN_SIZE, n_categories, num_layers=NUM_LAYERS)
    model.load_state_dict(torch.load(model_fn, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode once

    # Cache metadata to avoid recomputation
    metadata = {
        "categories": all_categories,
        "max_name": max_name,
        "vocab": vocab,
        "all_letters": all_letters,
        "n_letters": n_letters,
        "oob": oob,
        "vocab_size": vocab_size,
        "device": device,
        "loaded_at": time.time(),
    }

    return model, vectorizer, metadata


def _evict_old_models():
    """Remove oldest models if cache is full."""
    if len(_MODEL_CACHE) > MAX_CACHED_MODELS:
        # Find oldest model by load time
        oldest_key = min(
            _MODEL_CACHE.keys(), key=lambda k: _MODEL_CACHE[k][2]["loaded_at"]
        )
        del _MODEL_CACHE[oldest_key]


def _get_cached_model(
    model_fn: str, vocab_fn: str
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Thread-safe model retrieval with lazy loading."""
    global _MODEL_CACHE, _CACHE_LOCK, _CACHE_STATS

    if not CACHE_ENABLED:
        # If caching disabled, load directly
        return _load_and_cache_model(model_fn, vocab_fn)

    cache_key = _get_cache_key(model_fn, vocab_fn)

    # Check cache first (read lock)
    with _CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            _CACHE_STATS["hits"] += 1
            return _MODEL_CACHE[cache_key]

        _CACHE_STATS["misses"] += 1

    # Load model outside lock to avoid blocking other threads
    model, vectorizer, metadata = _load_and_cache_model(model_fn, vocab_fn)

    # Store in cache (write lock)
    with _CACHE_LOCK:
        _evict_old_models()  # Make room if needed
        _MODEL_CACHE[cache_key] = (model, vectorizer, metadata)
        _CACHE_STATS["loads"] += 1

    return model, vectorizer, metadata


def clear_model_cache(model_pattern: str | None = None) -> int:
    """Clear cached models matching pattern."""
    global _MODEL_CACHE, _CACHE_LOCK

    with _CACHE_LOCK:
        if model_pattern is None:
            count = len(_MODEL_CACHE)
            _MODEL_CACHE.clear()
            return count
        else:
            # Clear specific models matching pattern
            keys_to_remove = [k for k in _MODEL_CACHE.keys() if model_pattern in k]
            for key in keys_to_remove:
                del _MODEL_CACHE[key]
            return len(keys_to_remove)


def get_cache_info() -> dict[str, Any]:
    """Get detailed cache information."""
    with _CACHE_LOCK:
        return {
            "cache_enabled": CACHE_ENABLED,
            "cached_models": len(_MODEL_CACHE),
            "max_cached_models": MAX_CACHED_MODELS,
            "cache_stats": _CACHE_STATS.copy(),
            "models": list(_MODEL_CACHE.keys()),
        }


# Model parameter constants
MAX_NAME_FULLNAME = 47
MAX_NAME_FLORIDA = 30
MAX_NAME_CENSUS = 15
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS = 2


class EthnicolrModelClass:
    vocab: list[str] | None = None
    race: list[str] | None = None
    model: torch.nn.Module | None = None
    model_year: int | None = None

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
            raise ValueError(
                f"Column '{col}' not found in DataFrame. Available columns: {list(df.columns)}"
            )

        df.dropna(subset=[col], inplace=True)
        if df.shape[0] == 0:
            raise ValueError(f"Column '{col}' contains no non-NaN values.")

        df.drop_duplicates(subset=[col], inplace=True)

        return df

    @staticmethod
    def lineToTensor(
        line: str, all_letters: str, max_name: int, oob: int
    ) -> torch.Tensor:
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
        # Get file paths
        try:
            # Use modern importlib.resources for Python >= 3.9
            import ethnicolr2

            MODEL = str(files(ethnicolr2).joinpath(model_fn))
            VOCAB = str(files(ethnicolr2).joinpath(vocab_fn))
        except (NameError, AttributeError):
            # Fallback to pkg_resources for older Python versions
            MODEL = resource_filename(__name__, model_fn)
            VOCAB = resource_filename(__name__, vocab_fn)

        # Use cached model instead of loading every time
        model, vectorizer, model_metadata = _get_cached_model(MODEL, VOCAB)

        # Extract cached metadata to avoid recomputation
        all_categories = model_metadata["categories"]
        max_name = model_metadata["max_name"]
        all_letters = model_metadata["all_letters"]
        oob = model_metadata["oob"]
        device = model_metadata["device"]

        batch_size = BATCH_SIZE

        dataset = EthniDataset(
            df, all_letters, max_name, oob, transform=EthnicolrModelClass.lineToTensor
        )  # noqa
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Model is already loaded, configured, and in eval mode

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
                    [
                        dict(zip(all_categories, p, strict=False))
                        for p in probs.cpu().numpy()
                    ]
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
