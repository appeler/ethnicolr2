#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from typing import List, Optional

import pandas as pd

from .ethnicolr_class import EthnicolrModelClass
from .utils import arg_parser


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


def main(argv: Optional[List[str]] = None) -> None:
    """Main method for the command line interface."""
    if argv is None:
        argv = sys.argv[1:]
    args = arg_parser(
        argv,
        title="Predict Race/ethnicity by full name using the Florida voter registration data model.",
        default_out="pred_fl_reg_full_name.csv",
        first=True,
        full_name=True,
    )
    df = pd.read_csv(args.input, encoding="utf-8")
    rdf = pred_fl_full_name(
        df=df,
        full_name_col=args.full_name_col,
        lname_col=args.lname_col,
        fname_col=args.fname_col,
    )
    print(f"Writing output to {args.output}")
    rdf.to_csv(args.output, index=False)


if __name__ == "__main__":
    sys.exit(main())
