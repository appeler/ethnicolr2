#!/usr/bin/env python

"""
Tests for census_ln.py

"""

import unittest

import pandas as pd

from ethnicolr2.census_ln import census_ln


class TestCensusLn(unittest.TestCase):
    def setUp(self):
        names = [
            {"last": "smith", "true_race": "white"},
            {"last": "zhang", "true_race": "api"},
        ]
        self.df = pd.DataFrame(names)

    def tearDown(self):
        pass

    def test_census_ln_2000(self):
        odf = census_ln(self.df, "last", 2000)
        self.assertIn("pctwhite", odf.columns)
        self.assertEqual(odf.loc[odf["last"] == "smith", "pctwhite"].item(), 73.35)
        self.assertTrue(pd.api.types.is_float_dtype(odf["pctwhite"]))

    def test_census_ln_2010(self):
        odf = census_ln(self.df, "last", 2010)
        self.assertIn("pcthispanic", odf.columns)
        self.assertEqual(odf.loc[odf["last"] == "smith", "pctwhite"].item(), 70.9)
        self.assertTrue(pd.api.types.is_float_dtype(odf["pctwhite"]))

    def test_preserves_rows_index_and_input_columns(self):
        df = pd.DataFrame(
            {"last": ["smith", "smith", None], "pctwhite": [-1.0, -1.0, -1.0]},
            index=[8, 3, 12],
        )
        original = df.copy(deep=True)

        result = census_ln(df, "last", 2010)

        self.assertEqual(result.index.tolist(), [8, 3, 12])
        self.assertEqual(result["pctwhite"].iloc[:2].tolist(), [70.9, 70.9])
        self.assertTrue(pd.isna(result.loc[12, "pctwhite"]))
        self.assertNotIn("pctwhite_x", result.columns)
        pd.testing.assert_frame_equal(df, original)


if __name__ == "__main__":
    unittest.main()
