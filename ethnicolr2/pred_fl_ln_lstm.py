#!/usr/bin/env python

import sys

import click
import pandas as pd

from .cli_utils import common_options, name_column_options, validate_input_file
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
            DataFrame with original data plus:
                - 'preds': Predicted race/ethnicity category
                - 'probs': Dictionary of probabilities for each category

        Raises:
            ValueError: If lname_col doesn't exist or DataFrame is invalid
            RuntimeError: If model prediction fails
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
        if not isinstance(lname_col, str):
            raise TypeError(f"Expected string for lname_col, got {type(lname_col)}")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")

        df["__name"] = df[lname_col].str.title()

        rdf = cls.predict(df=df, vocab_fn=cls.VOCAB_FN, model_fn=cls.MODEL_FN)

        del rdf["__name"]
        return rdf


pred_fl_last_name = LastNameLstmModel.pred_fl_last_name


@click.command()
@click.argument("input_file", callback=validate_input_file, metavar="INPUT_FILE")
@common_options
@name_column_options
def main(input_file: str, output: str, verbose: bool, last_name_col: str) -> None:
    """Predict race/ethnicity by last name using Florida voter registration model.

    INPUT_FILE: Path to CSV file containing name data.
    """
    if output is None:
        output = "pred_fl_reg_last_name.csv"

    if verbose:
        click.echo(f"Loading data from: {input_file}")
        click.echo(f"Last name column: {last_name_col}")
        click.echo("Using Florida voter registration LSTM model")

    try:
        df = pd.read_csv(input_file, encoding="utf-8")

        if verbose:
            click.echo(f"Loaded {len(df)} rows")

        rdf = pred_fl_last_name(df=df, lname_col=last_name_col)

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
