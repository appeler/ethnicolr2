"""Prefetch ethnicolr2 model assets into the Hugging Face cache."""

from __future__ import annotations

import click

from ._resources import HF_REPO, HF_REVISION, resolve_model

MODEL_FILES = (
    "lstm_lastname_gen.pt",
    "lstm_fullname.pt",
    "census_lstm_lastname.pt",
    "pt_vec_lastname.joblib",
    "pt_vec_fullname.joblib",
    "pt_vec_census_lastname.joblib",
)


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Show each resolved asset.")
def download_cli(verbose: bool = False) -> None:
    """Download the pinned model assets into the Hugging Face cache."""
    for filename in MODEL_FILES:
        path = resolve_model(filename)
        if verbose:
            click.echo(f"{filename}: {path}")
    click.echo(f"Resolved {len(MODEL_FILES)} assets from {HF_REPO}@{HF_REVISION}.")


if __name__ == "__main__":
    download_cli()
