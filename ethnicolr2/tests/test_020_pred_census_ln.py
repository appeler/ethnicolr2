#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for pred_census_last_name

"""

import os
import shutil
import unittest
import pandas as pd
from ethnicolr2.pred_census_last_name import pred_census_last_name
from . import capture

race = ["api", "black", "hispanic", "white"]
race_mean = ["api_mean", "black_mean", "hispanic_mean", "white_mean"]


class TestCensusLn(unittest.TestCase):
    def setUp(self):
        names = [
            {"last": "smith", "true_race": "white"},
            {"last": "zhang", "true_race": "api"},
        ]
        self.df = pd.DataFrame(names)

    def tearDown(self):
        pass

    def test_pred_census_ln_2010(self):
        odf = pred_census_last_name(self.df, "last", 2010)
        self.assertTrue(
            all(
                odf[[col for col in odf.columns
                     if col in race]].sum(axis=1).round(1)
                == 1.0
            )
        )
        self.assertTrue(all(odf.true_race == odf.race))

    def test_pred_census_ln_2010_mean(self):
        odf = pred_census_last_name(self.df, "last", 2010, conf_int=0.9)
        self.assertTrue(
            all(
                odf[[col for col in odf.columns
                     if col in race_mean]].sum(axis=1).round(1)
                == 1.0
            )
        )
        self.assertTrue(all(odf.true_race == odf.race))

if __name__ == "__main__":
    unittest.main()
