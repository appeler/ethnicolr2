"""Resolve immutable ethnicolr2 model assets."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import (
    hf_hub_download,  # pyright: ignore[reportUnknownVariableType]
)

HF_REPO = "gojiberries/ethnicolr2"
HF_REVISION = "31fa6f7a443d254c9d62bca2cbc29890fcc82bbb"
MODEL_DIR_ENV = "ETHNICOLR2_MODEL_DIR"


def resolve_model(filename: str) -> str:
    """Return a local path for a pinned model artifact.

    An explicit model directory is checked first. Otherwise, the immutable Hub
    asset is downloaded through the standard Hugging Face cache.
    """
    filename = filename.removeprefix("models/")
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        candidate = Path(override) / filename
        if candidate.is_file():
            return str(candidate)

    return hf_hub_download(  # pyright: ignore[reportUnknownVariableType]
        HF_REPO, filename, revision=HF_REVISION
    )
