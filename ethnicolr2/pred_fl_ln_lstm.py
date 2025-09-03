#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from typing import List, Optional

import pandas as pd

from .ethnicolr_class import EthnicolrModelClass
from .utils import arg_parser


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


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    args = arg_parser(
        argv,
        title="Predict Race/ethnicity by last name using the Florida voter registration data model.",
        default_out="pred_fl_reg_last_name.csv",
    )
    df = pd.read_csv(args.input, encoding="utf-8")
    rdf = pred_fl_last_name(df=df, lname_col=args.lname_col)
    print(f"Writing output to {args.output}")
    rdf.to_csv(args.output, index=False)


if __name__ == "__main__":
    sys.exit(main())
