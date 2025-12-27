#!/usr/bin/env python

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    import pandas as pd
else:
    import pandas as pd

from .cli_utils import common_options, full_name_options, validate_input_file
from .ethnicolr_class import EthnicolrModelClass


class FullNameLstmModel(EthnicolrModelClass):
    """Predict ethnicity based on fullname.

    Attributes:
        MODEL_FN: Path to the LSTM model file
        VOCAB_FN: Path to the vocabulary vectorizer file
    """

    MODEL_FN = "models/lstm_fullname.pt"
    VOCAB_FN = "models/pt_vec_fullname.joblib"

    @classmethod
    def pred_fl_full_name(
        cls,
        df: pd.DataFrame,
        full_name_col: str | None = None,
        lname_col: str | None = None,
        fname_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Predict the race/ethnicity by the full name using the Florida voter
        registration data model.

        Args:
            df: Pandas DataFrame containing name columns
            full_name_col: Column name for the full name (optional)
            lname_col: Column name for the last name (optional)
            fname_col: Column name for the first name (optional)

        Returns:
            pd.DataFrame: Pandas DataFrame with additional columns:
                - `preds` the prediction result
                - `probs` probability dictionary for each category

        Raises:
            ValueError: If column arguments are invalid or missing

        """

        match (bool(lname_col and fname_col), bool(full_name_col)):
            case (True, _):
                if lname_col not in df.columns:
                    raise ValueError(f"Column '{lname_col}' not found in DataFrame")
                if fname_col not in df.columns:
                    raise ValueError(f"Column '{fname_col}' not found in DataFrame")
                df["__name"] = (
                    df[lname_col].str.strip() + " " + df[fname_col].str.strip()
                ).str.title()
            case (False, True):
                if full_name_col not in df.columns:
                    raise ValueError(f"Column '{full_name_col}' not found in DataFrame")
                df["__name"] = df[full_name_col].str.title()
            case _:
                raise ValueError(
                    "Must provide either full_name_col or both lname_col and fname_col"
                )

        rdf = cls.predict(df, cls.VOCAB_FN, cls.MODEL_FN)

        rdf = rdf.drop(columns=["__name"])
        return rdf


def pred_fl_full_name(
    df: pd.DataFrame,
    full_name_col: str | None = None,
    lname_col: str | None = None,
    fname_col: str | None = None,
) -> pd.DataFrame:
    """Predict race/ethnicity by full name using Florida voter registration model.

    Args:
        df: Pandas DataFrame containing name columns
        full_name_col: Column name for full name (optional)
        lname_col: Column name for last name (optional)
        fname_col: Column name for first name (optional)

    Returns:
        pd.DataFrame: DataFrame with predictions and probabilities
    """
    return FullNameLstmModel.pred_fl_full_name(df, full_name_col, lname_col, fname_col)


@click.command()
@click.argument("input_file", callback=validate_input_file, metavar="INPUT_FILE")
@common_options
@full_name_options
def main(
    input_file: str,
    output: str | None,
    verbose: bool,
    full_name_col: str,
    first_name_col: str,
    last_name_col: str,
) -> None:
    """Predict race/ethnicity by full name using Florida voter registration model.

    Args:
        input_file: Path to CSV file containing name data
        output: Output file path
        verbose: Enable verbose output
        full_name_col: Column name containing full names (optional)
        first_name_col: Column name containing first names (optional)
        last_name_col: Column name containing last names (optional)

    INPUT_FILE: Path to CSV file containing name data.

    You must provide either:
    - A full name column (--full-name-col)
    - Both first and last name columns (--first-name-col and --last-name-col)
    """
    if output is None:
        output = "pred_fl_reg_full_name.csv"

    # Validate that we have the required name columns
    if not full_name_col and not (first_name_col and last_name_col):
        click.echo(
            "Error: Must provide either --full-name-col or both --first-name-col and --last-name-col",
            err=True,
        )
        sys.exit(1)

    if verbose:
        click.echo(f"Loading data from: {input_file}")
        if full_name_col:
            click.echo(f"Full name column: {full_name_col}")
        else:
            click.echo(f"First name column: {first_name_col}")
            click.echo(f"Last name column: {last_name_col}")
        click.echo("Using Florida voter registration full name LSTM model")

    try:
        df = pd.read_csv(input_file, encoding="utf-8")  # type: ignore[misc]

        if verbose:
            click.echo(f"Loaded {len(df)} rows")

        rdf = pred_fl_full_name(
            df=df,
            full_name_col=full_name_col,
            lname_col=last_name_col,
            fname_col=first_name_col,
        )

        rdf.to_csv(output, index=False)

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


if __name__ == "__main__":
    main()
