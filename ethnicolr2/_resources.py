"""Resolve immutable ethnicolr2 model assets."""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

from huggingface_hub import (
    hf_hub_download,  # pyright: ignore[reportUnknownVariableType]
)

HF_REPO = "gojiberries/ethnicolr2"
HF_REVISION = "0ba3137f834a50482f3130e92168939e9a2de889"
MODEL_DIR_ENV = "ETHNICOLR2_MODEL_DIR"


def resolve_model(filename: str) -> str:
    """Return a local path for a pinned model artifact.

    An explicit model directory is checked first, followed by a development
    checkout's bundled file. Clean installs download the immutable Hub asset.
    """
    filename = filename.removeprefix("models/")
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        candidate = Path(override) / filename
        if candidate.is_file():
            return str(candidate)

    bundled = Path(str(files("ethnicolr2") / "models" / filename))
    if bundled.is_file():
        return str(bundled)

    return hf_hub_download(  # pyright: ignore[reportUnknownVariableType]
        HF_REPO, filename, revision=HF_REVISION
    )
