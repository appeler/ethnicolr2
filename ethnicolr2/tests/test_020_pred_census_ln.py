#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for pred_census_last_name

"""

import os
import shutil
import unittest
import pandas as pd
from ethnicolr2 import pred_census_last_name
from . import capture

race = ["api", "black", "hispanic", "white"]
race_mean = ["api_mean", "black_mean", "hispanic_mean", "white_mean"]


class TestCensusLn(unittest.TestCase):
    def setUp(self):
        names = [
            {"last": "smith", "true_race": "nh_white"},
            {"last": "zhang", "true_race": "asian"},
        ]
        self.df = pd.DataFrame(names)

    def tearDown(self):
        pass

    def test_pred_census_ln_2010(self):
        odf = pred_census_last_name(self.df, "last")
        self.assertTrue(all(odf.true_race == odf.preds))

if __name__ == "__main__":
    unittest.main()
