"""Click-based CLI utilities for ethnicolr2."""

from pathlib import Path

import click


def validate_input_file(ctx, param, value):
    """Validate that the input file exists."""
    if value is None:
        return value

    path = Path(value)
    if not path.exists():
        raise click.BadParameter(f"Input file '{value}' does not exist.")
    if not path.is_file():
        raise click.BadParameter(f"'{value}' is not a file.")
    return str(path)


def validate_output_file(ctx, param, value):
    """Validate the output file path."""
    if value is None:
        return value

    path = Path(value)
    # Check if parent directory exists
    if not path.parent.exists():
        raise click.BadParameter(f"Output directory '{path.parent}' does not exist.")
    return str(path)


def common_options(func):
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


def name_column_options(func):
    """Decorator to add name column options."""
    func = click.option(
        "--last-name-col",
        "-l",
        help="Column name containing last names.",
        required=True,
        metavar="COLUMN",
    )(func)
    return func


def full_name_options(func):
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


def prediction_options(func):
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


def year_option(years: list, default_year: int):
    """Factory function to create year option with specific choices."""

    def decorator(func):
        return click.option(
            "--year",
            "-y",
            type=click.Choice([str(y) for y in years]),
            default=str(default_year),
            help="Census data year to use.",
            show_default=True,
        )(func)

    return decorator
