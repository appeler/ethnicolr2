#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for pytorch models
"""

import unittest
import pandas as pd
from ethnicolr2 import pred_fl_full_name
from ethnicolr2 import pred_fl_last_name
from ethnicolr2 import pred_census_last_name


class TestPytorchModels(unittest.TestCase):
    def setUp(self):
        names = [
            {"last": "hernandez", "first": "hector"},
            {"last": "zhang", "first": "simon"},
        ]
        self.df = pd.DataFrame(names)

    def tearDown(self):
        pass

    def test_fullname_v1(self):
        odf = pred_fl_full_name(self.df, lname_col="last", fname_col="first")
        self.assertEqual(odf[odf["last"] == "hernandez"]["preds"].values[0], "hispanic")
        self.assertEqual(odf[odf["last"] == "zhang"]["preds"].values[0], "asian")

    def test_fullname_v2(self):
        self.df["fullname"] = self.df["last"] + " " + self.df["first"]
        odf = pred_fl_full_name(self.df, full_name_col="fullname")
        pd.set_option("display.max_colwidth", None)
        print(odf)
        self.assertEqual(odf[odf["last"] == "hernandez"]["preds"].values[0], "hispanic")
        self.assertEqual(odf[odf["last"] == "zhang"]["preds"].values[0], "asian")

    def test_lastname(self):
        odf = pred_fl_last_name(self.df, lname_col="last")
        self.assertEqual(odf[odf["last"] == "hernandez"]["preds"].values[0], "hispanic")
        self.assertEqual(odf[odf["last"] == "zhang"]["preds"].values[0], "asian")

    def test_census_lastname(self):
        odf = pred_census_last_name(self.df, lname_col="last")
        self.assertEqual(odf[odf["last"] == "hernandez"]["preds"].values[0], "hispanic")
        self.assertEqual(odf[odf["last"] == "zhang"]["preds"].values[0], "asian")


if __name__ == "__main__":
    unittest.main()
