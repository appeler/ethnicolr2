"""Click-based CLI utilities for ethnicolr2."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import pandas as pd


def validate_input_file(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> str | None:
    """Validate that the input file exists.

    Args:
        ctx: Click context object
        param: Click parameter object
        value: File path to validate

    Returns:
        str | None: Validated file path or None

    Raises:
        click.BadParameter: If file doesn't exist or is not a file
    """
    if value is None:
        return value

    path = Path(value)
    if not path.exists():
        raise click.BadParameter(f"Input file '{value}' does not exist.")
    if not path.is_file():
        raise click.BadParameter(f"'{value}' is not a file.")
    return str(path)


def validate_output_file(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> str | None:
    """Validate the output file path."""
    if value is None:
        return value

    path = Path(value)
    # Check if parent directory exists
    if not path.parent.exists():
        raise click.BadParameter(f"Output directory '{path.parent}' does not exist.")
    return str(path)


def common_options(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to add common CLI options to all commands."""
    func = click.option(
        "--output",
        "-o",
        help="Output CSV file path.",
        callback=validate_output_file,
        metavar="PATH",
    )(func)
    func = click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")(
        func
    )
    return func


def name_column_options(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to add name column options."""
    func = click.option(
        "--last-name-col",
        "-l",
        help="Column name containing last names.",
        required=True,
        metavar="COLUMN",
    )(func)
    return func


def full_name_options(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to add full name options."""
    func = click.option(
        "--full-name-col",
        help="Column name containing full names (alternative to first/last name columns).",
        metavar="COLUMN",
    )(func)
    func = click.option(
        "--first-name-col",
        "-f",
        help="Column name containing first names.",
        metavar="COLUMN",
    )(func)
    func = click.option(
        "--last-name-col",
        "-l",
        help="Column name containing last names.",
        metavar="COLUMN",
    )(func)
    return func


def prediction_options(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to add prediction-related options."""
    func = click.option(
        "--iterations",
        "-i",
        default=100,
        help="Number of iterations to measure uncertainty.",
        show_default=True,
        metavar="N",
    )(func)
    func = click.option(
        "--confidence",
        "-c",
        default=1.0,
        help="Confidence interval for predictions.",
        show_default=True,
        metavar="FLOAT",
    )(func)
    return func


def year_option(
    years: list[int], default_year: int
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Factory function to create year option with specific choices."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return click.option(
            "--year",
            "-y",
            type=click.Choice([str(y) for y in years]),
            default=str(default_year),
            help="Census data year to use.",
            show_default=True,
        )(func)

    return decorator


def run_prediction(
    input_file: str,
    output: str,
    verbose: bool,
    predict: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    """Read a CSV, run one prediction over it, and write the result.

    Errors are reported on stderr and end the process with status 1, so every
    command fails the same way.

    Args:
        input_file: CSV file to read.
        output: CSV file to write.
        verbose: Report row counts and the output path in more detail.
        predict: Turns the input rows into the rows to write.
    """
    try:
        df = pd.read_csv(input_file, encoding="utf-8")  # type: ignore[misc]

        if verbose:
            click.echo(f"Loaded {len(df)} rows")

        predict(df).to_csv(output, index=False)

        if verbose:
            click.echo(f"Predictions saved to: {output}")
        else:
            click.echo(f"Writing output to {output}")

    except FileNotFoundError:
        click.echo(f"Error: File '{input_file}' not found.", err=True)
        sys.exit(1)
    except KeyError as e:
        click.echo(f"Error: Column {e} not found in input file.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
