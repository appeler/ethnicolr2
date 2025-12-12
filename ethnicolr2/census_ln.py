#!/usr/bin/env python

import sys
from importlib.resources import files

import click
import pandas as pd

import ethnicolr2

from .cli_utils import (
    common_options,
    name_column_options,
    validate_input_file,
    year_option,
)
from .ethnicolr_class import EthnicolrModelClass

CENSUS2000 = str(files(ethnicolr2).joinpath("data/census/census_2000.csv"))
CENSUS2010 = str(files(ethnicolr2).joinpath("data/census/census_2010.csv"))

CENSUS_COLS = ["pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic"]


class CensusLnData:
    census_df = None
    census_year = None

    @classmethod
    def census_ln(
        cls, df: pd.DataFrame, lname_col: str, year: int = 2000
    ) -> pd.DataFrame:
        """Appends columns from Census data to the input DataFrame
        based on the last name.

        Removes extra space. Checks if the name is the Census data.  If it is,
        outputs data from that row.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the first and last name
                columns.
            lname_col (str): Column name for the last name.
            year (int): The year of Census data to be used. (2000 or 2010)
                (default is 2000)

        Returns:
            DataFrame: Pandas DataFrame with additional columns 'pctwhite',
                'pctblack', 'pctapi', 'pctaian', 'pct2prace', 'pcthispanic'

        """

        df = EthnicolrModelClass.test_and_norm_df(df, lname_col)

        df["__last_name"] = df[lname_col].str.strip().str.upper()

        if cls.census_df is None or cls.census_year != year:
            match year:
                case 2000:
                    cls.census_df = pd.read_csv(
                        CENSUS2000, usecols=["name"] + CENSUS_COLS
                    )
                case 2010:
                    cls.census_df = pd.read_csv(
                        CENSUS2010, usecols=["name"] + CENSUS_COLS
                    )
                case _:
                    raise ValueError(
                        f"Unsupported census year: {year}. Only 2000 and 2010 are supported."
                    )

            cls.census_df.drop(
                cls.census_df[cls.census_df.name.isnull()].index, inplace=True
            )

            cls.census_df.columns = ["__last_name"] + CENSUS_COLS
            cls.census_year = year

        rdf = pd.merge(df, cls.census_df, how="left", on="__last_name")

        del df["__last_name"]
        del rdf["__last_name"]

        return rdf


census_ln = CensusLnData.census_ln


@click.command()
@click.argument("input_file", callback=validate_input_file, metavar="INPUT_FILE")
@common_options
@name_column_options
@year_option(years=[2000, 2010], default_year=2010)
def main(
    input_file: str, output: str, verbose: bool, last_name_col: str, year: str
) -> None:
    """Append Census demographic data by last name.

    INPUT_FILE: Path to CSV file containing name data.
    """
    if output is None:
        output = "census-output.csv"

    year_int = int(year)

    if verbose:
        click.echo(f"Loading data from: {input_file}")
        click.echo(f"Using Census {year} data")
        click.echo(f"Last name column: {last_name_col}")

    try:
        df = pd.read_csv(input_file)

        if verbose:
            click.echo(f"Loaded {len(df)} rows")

        rdf = census_ln(df, last_name_col, year_int)

        rdf.to_csv(output, index=False)

        if verbose:
            click.echo(f"Results saved to: {output}")
        else:
            click.echo(f"Saving output to file: `{output}`")

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
