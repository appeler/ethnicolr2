#!/usr/bin/env python

from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    import pandas as pd
else:
    import pandas as pd

from .cli_utils import (
    common_options,
    name_column_options,
    run_prediction,
    validate_input_file,
)
from .ethnicolr_class import EthnicolrModelClass


class LastNameLstmModel(EthnicolrModelClass):
    MODEL_FN = "models/lstm_lastname_gen.pt"
    VOCAB_FN = "models/pt_vec_lastname.joblib"

    @classmethod
    def pred_fl_last_name(cls, df: pd.DataFrame, lname_col: str) -> pd.DataFrame:
        """
        Predict the race/ethnicity by the last name using the Florida voter
        registration data model.

        Args:
            df: Pandas DataFrame containing the last name column
            lname_col: Column name for the last name

        Returns:
            pd.DataFrame: DataFrame with original data plus:
                - 'preds': Predicted race/ethnicity category
                - 'probs': Dictionary of probabilities for each category

        Raises:
            ValueError: If lname_col doesn't exist or DataFrame is invalid
            TypeError: If inputs are not correct types
        """
        if not hasattr(df, "columns") or not hasattr(df, "empty"):
            raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        if lname_col not in df.columns:
            raise ValueError(f"Column '{lname_col}' not found in DataFrame")

        work_df = df.copy()
        work_df["__name"] = work_df[lname_col].str.title()

        rdf = cls.predict(df=work_df, vocab_fn=cls.VOCAB_FN, model_fn=cls.MODEL_FN)

        rdf = rdf.drop(columns=["__name"])
        return rdf


def pred_fl_last_name(df: pd.DataFrame, lname_col: str) -> pd.DataFrame:
    """Predict race/ethnicity by last name using Florida voter registration model.

    Args:
        df: Pandas DataFrame containing the last name column
        lname_col: Column name for the last name

    Returns:
        pd.DataFrame: DataFrame with predictions and probabilities
    """
    return LastNameLstmModel.pred_fl_last_name(df, lname_col)


@click.command()
@click.argument("input_file", callback=validate_input_file, metavar="INPUT_FILE")
@common_options
@name_column_options
def main(
    input_file: str, output: str | None, verbose: bool, last_name_col: str
) -> None:
    """Predict race/ethnicity by last name using Florida voter registration model.

    Args:
        input_file: Path to CSV file containing name data
        output: Output file path
        verbose: Enable verbose output
        last_name_col: Column name containing last names

    INPUT_FILE: Path to CSV file containing name data.
    """
    if output is None:
        output = "pred_fl_reg_last_name.csv"

    if verbose:
        click.echo(f"Loading data from: {input_file}")
        click.echo(f"Last name column: {last_name_col}")
        click.echo("Using Florida voter registration LSTM model")

    run_prediction(
        input_file,
        output,
        verbose,
        lambda df: pred_fl_last_name(df=df, lname_col=last_name_col),
    )


if __name__ == "__main__":
    main()
