#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd

from .ethnicolr_class import EthnicolrModelClass
from .utils import arg_parser


class CensusLastNameLstmModel(EthnicolrModelClass):
    MODEL_FN = "models/census_lstm_lastname.pt"
    VOCAB_FN = "models/pt_vec_census_lastname.joblib"

    @classmethod
    def pred_fl_last_name(cls,
                          df: pd.DataFrame,
                          lname_col: str) -> pd.DataFrame:
        """
        Predict the race/ethnicity by the last name using the Florida voter
        registration data model.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the first and last name
                columns.
            lname_col (str): Column name for the last name.

        Returns:
            DataFrame: Pandas DataFrame with additional columns:
                - `race` the predict result
                - Additional columns for probability of each classes.

        """

        df['__name'] = df[lname_col].str.title()

        rdf = cls.predict(df=df,
                            vocab_fn=cls.VOCAB_FN,
                            model_fn=cls.MODEL_FN)

        del rdf['__name']
        return rdf


pred_census_last_name = CensusLastNameLstmModel.pred_fl_last_name

def main(argv=sys.argv[1:]) -> None:
    args = arg_parser(argv,
                      title = "Predict Race/ethnicity by last name using the Florida voter registration data model.",
                      default_out="pred_fl_reg_last_name.csv",
    )
    df = pd.read_csv(args.input, encoding="utf-8")
    rdf = pred_census_last_name(df=df, lname_col=args.lname_col)
    print(f"Writing output to {args.output}")
    rdf.to_csv(args.output, index=False)


if __name__ == "__main__":
    sys.exit(main())
