#!/usr/bin/env python

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    import pandas as pd
else:
    import pandas as pd

from .cli_utils import common_options, name_column_options, validate_input_file
from .ethnicolr_class import EthnicolrModelClass


class CensusLastNameLstmModel(EthnicolrModelClass):
    MODEL_FN = "models/census_lstm_lastname.pt"
    VOCAB_FN = "models/pt_vec_census_lastname.joblib"

    @classmethod
    def pred_census_last_name(cls, df: pd.DataFrame, lname_col: str) -> pd.DataFrame:
        """
        Predict the race/ethnicity by the last name using the Census
        data model.

        Args:
            df: Pandas DataFrame containing the first and last name columns
            lname_col: Column name for the last name

        Returns:
            pd.DataFrame: Pandas DataFrame with additional columns:
                - `preds` the prediction result
                - `probs` probability dictionary for each category

        """

        df["__name"] = df[lname_col].str.title()

        rdf = cls.predict(df=df, vocab_fn=cls.VOCAB_FN, model_fn=cls.MODEL_FN)

        rdf = rdf.drop(columns=["__name"])
        return rdf


def pred_census_last_name(df: pd.DataFrame, lname_col: str) -> pd.DataFrame:
    """Predict race/ethnicity by last name using Census LSTM model.

    Args:
        df: Pandas DataFrame containing the last name column
        lname_col: Column name for the last name

    Returns:
        pd.DataFrame: DataFrame with predictions and probabilities
    """
    return CensusLastNameLstmModel.pred_census_last_name(df, lname_col)


@click.command()
@click.argument("input_file", callback=validate_input_file, metavar="INPUT_FILE")
@common_options
@name_column_options
def main(
    input_file: str, output: str | None, verbose: bool, last_name_col: str
) -> None:
    """Predict race/ethnicity by last name using Census LSTM model.

    Args:
        input_file: Path to CSV file containing name data
        output: Output file path
        verbose: Enable verbose output
        last_name_col: Column name containing last names

    INPUT_FILE: Path to CSV file containing name data.
    """
    if output is None:
        output = "pred_census_last_name.csv"

    if verbose:
        click.echo(f"Loading data from: {input_file}")
        click.echo(f"Last name column: {last_name_col}")
        click.echo("Using Census LSTM model")

    try:
        df = pd.read_csv(input_file, encoding="utf-8")  # type: ignore[misc]

        if verbose:
            click.echo(f"Loaded {len(df)} rows")

        rdf = pred_census_last_name(df=df, lname_col=last_name_col)

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
