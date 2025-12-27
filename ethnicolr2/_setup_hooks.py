"""Setup hooks and utilities for ethnicolr2."""

import urllib.error
import urllib.request
from importlib.resources import files
from pathlib import Path

import click

# Model URLs - these would need to be updated with actual model URLs
MODEL_URLS = {
    "lstm_lastname_gen.pt": "https://github.com/appeler/ethnicolr2/releases/download/models/lstm_lastname_gen.pt",
    "lstm_fullname.pt": "https://github.com/appeler/ethnicolr2/releases/download/models/lstm_fullname.pt",
    "census_lstm_lastname.pt": "https://github.com/appeler/ethnicolr2/releases/download/models/census_lstm_lastname.pt",
    "pt_vec_lastname.joblib": "https://github.com/appeler/ethnicolr2/releases/download/models/pt_vec_lastname.joblib",
    "pt_vec_fullname.joblib": "https://github.com/appeler/ethnicolr2/releases/download/models/pt_vec_fullname.joblib",
    "pt_vec_census_lastname.joblib": "https://github.com/appeler/ethnicolr2/releases/download/models/pt_vec_census_lastname.joblib",
}


def get_models_directory() -> Path:
    """Get the models directory path.

    Returns:
        Path: Path to the models directory
    """
    import ethnicolr2

    models_resource = files(ethnicolr2) / "models"
    return Path(str(models_resource))


def download_model(
    model_name: str, url: str, models_dir: Path, verbose: bool = False
) -> bool:
    """Download a single model file.

    Args:
        model_name: Name of the model file to download
        models_dir: Directory to save the model
        url: URL to download the model from
        verbose: Enable verbose output

    Returns:
        bool: True if download successful, False otherwise
    """
    model_path = models_dir / model_name

    # Check if model already exists
    if model_path.exists():
        if verbose:
            click.echo(f"Model {model_name} already exists, skipping download.")
        return True

    if verbose:
        click.echo(f"Downloading {model_name}...")

    try:
        # Create directory if it doesn't exist
        models_dir.mkdir(parents=True, exist_ok=True)

        # Download the file
        urllib.request.urlretrieve(url, model_path)

        if verbose:
            click.echo(f"Successfully downloaded {model_name}")
        return True

    except urllib.error.URLError as e:
        click.echo(f"Error downloading {model_name}: {e}", err=True)
        return False
    except Exception as e:
        click.echo(f"Unexpected error downloading {model_name}: {e}", err=True)
        return False


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.option(
    "--force", "-f", is_flag=True, help="Force re-download even if files exist."
)
def download_cli(verbose: bool = False, force: bool = False) -> None:
    """Download pre-trained models for ethnicolr2.

    Args:
        verbose: Enable verbose output
        force: Force re-download even if files exist

    This command downloads the LSTM models and vectorizers needed for
    ethnicity prediction. Models are downloaded to the package models directory.
    """
    models_dir = get_models_directory()

    if verbose:
        click.echo(f"Models directory: {models_dir}")

    if force:
        if verbose:
            click.echo("Force mode enabled - will overwrite existing models")
        # Remove existing models if force is enabled
        for model_name in MODEL_URLS.keys():
            model_path = models_dir / model_name
            if model_path.exists():
                model_path.unlink()
                if verbose:
                    click.echo(f"Removed existing {model_name}")

    success_count = 0
    total_models = len(MODEL_URLS)

    for model_name, url in MODEL_URLS.items():
        if download_model(model_name, url, models_dir, verbose):
            success_count += 1

    if success_count == total_models:
        click.echo(f"Successfully downloaded all {total_models} model files.")
    else:
        click.echo(
            f"Downloaded {success_count}/{total_models} model files. Some downloads may have failed.",
            err=True,
        )
        if success_count < total_models:
            exit(1)


if __name__ == "__main__":
    download_cli()
