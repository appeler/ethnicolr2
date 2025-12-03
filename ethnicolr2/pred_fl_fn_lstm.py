#!/usr/bin/env python

import sys

import click
import pandas as pd

from .cli_utils import common_options, full_name_options, validate_input_file
from .ethnicolr_class import EthnicolrModelClass


class FullNameLstmModel(EthnicolrModelClass):
    """Predict ethnicity based on fullname"""

    MODEL_FN = "models/lstm_fullname.pt"
    VOCAB_FN = "models/pt_vec_fullname.joblib"

    @classmethod
    def pred_fl_full_name(
        cls,
        df: pd.DataFrame,
        full_name_col: str = None,
        lname_col: str = None,
        fname_col: str = None,
    ) -> pd.DataFrame:
        """
        Predict the race/ethnicity by the full name using the Florida voter
        registration data model.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the first and last name
                columns.
            ful_name_col (str): Column name for the full name.
            lname_col (str): Column name for the last name.
            fname_col (str or int): Column name for the first name.

        Returns:
            DataFrame: Pandas DataFrame with additional columns:
                - `race` the prediction result
                - Additional columns for the probability of each of the classes.

        """

        if lname_col and fname_col:
            if lname_col not in df.columns:
                raise ValueError(f"Column '{lname_col}' not found in DataFrame")
            if fname_col not in df.columns:
                raise ValueError(f"Column '{fname_col}' not found in DataFrame")
            df["__name"] = (
                df[lname_col].str.strip() + " " + df[fname_col].str.strip()
            ).str.title()
        elif full_name_col:
            if full_name_col not in df.columns:
                raise ValueError(f"Column '{full_name_col}' not found in DataFrame")
            df["__name"] = df[full_name_col].str.title()
        else:
            raise ValueError(
                "Must provide either full_name_col or both lname_col and fname_col"
            )

        rdf = cls.predict(df, cls.VOCAB_FN, cls.MODEL_FN)

        del rdf["__name"]
        return rdf


pred_fl_full_name = FullNameLstmModel.pred_fl_full_name


@click.command()
@click.argument("input_file", callback=validate_input_file, metavar="INPUT_FILE")
@common_options
@full_name_options
def main(
    input_file: str,
    output: str,
    verbose: bool,
    full_name_col: str,
    first_name_col: str,
    last_name_col: str,
) -> None:
    """Predict race/ethnicity by full name using Florida voter registration model.

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
        df = pd.read_csv(input_file, encoding="utf-8")

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
