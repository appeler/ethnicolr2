from __future__ import annotations

import argparse


def arg_parser(
    argv: list[str],
    title: str,
    default_out: str,
    default_year: int = 2000,
    year_choices: list[int] | None = None,
    first: bool = False,
    full_name: bool = False,
) -> argparse.Namespace:
    """Parse command line arguments for ethnicolr2 CLI tools.

    Args:
        argv: Command line argument list
        title: Description for the argument parser
        default_out: Default output filename
        default_year: Default year choice
        year_choices: Available year options
        first: Whether to include first name argument
        full_name: Whether to include full name argument

    Returns:
        argparse.Namespace: Parsed command line arguments

    Raises:
        TypeError: If argv is not a list
    """
    if not hasattr(argv, "__iter__") or isinstance(argv, str):
        raise TypeError(f"Expected list for argv, got {type(argv)}")

    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("input", default=None, help="Input file")
    parser.add_argument(
        "-o", "--output", default=default_out, help="Output file with prediction data"
    )
    if first:
        parser.add_argument(
            "-f",
            "--first",
            "--fname_col",
            dest="fname_col",
            required=True,
            help="Column name for the column with the first name",
        )
    parser.add_argument(
        "-l",
        "--last",
        "--lname_col",
        dest="lname_col",
        required=not full_name,
        help="Column name for the column with the last name",
    )
    if full_name:
        parser.add_argument(
            "--full_name_col",
            help="Column name for the full name column",
        )
    parser.add_argument(
        "-i",
        "--iter",
        default=100,
        type=int,
        help="Number of iterations to measure uncertainty",
    )
    parser.add_argument(
        "-c",
        "--conf",
        default=1.0,
        type=float,
        help="Confidence interval of Predictions",
    )
    if year_choices is not None:
        parser.add_argument(
            "-y",
            "--year",
            type=int,
            default=default_year,
            choices=year_choices,
            help=f"Year of data (default={default_year})",
        )
    args = parser.parse_args(argv)

    print(args)

    return args
