#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for FL voter registration models

"""

import os
import shutil
import unittest
import pandas as pd
from ethnicolr2 import pred_fl_last_name
from ethnicolr2 import pred_fl_full_name
from pkg_resources import resource_filename

from . import capture

race = ["asian", "hispanic", "nh_black", "nh_white"]
race5 = ["asian", "hispanic", "nh_black", "nh_white", "other"]

race_mean = ["asian_mean", "hispanic_mean", "nh_black_mean", "nh_white_mean"]
race5_mean = ["asian_mean", "hispanic_mean", "nh_black_mean", "nh_white_mean",
              "other_mean"]


class TestPredFL(unittest.TestCase):
    def setUp(self):
        names = [
            {"last": "zhang", "first": "simon", "true_race": "asian"},
            {"last": "torres", "first": "raul", "true_race": "hispanic"},
        ]
        self.df = pd.DataFrame(names)

    def tearDown(self):
        pass

    def test_pred_fl_reg_ln(self):
        odf = pred_fl_full_name(self.df, "last")
        self.assertTrue(all(odf.true_race == odf.preds))

    def test_pred_fl_reg_name(self):
        odf = pred_fl_full_name(self.df, "last", "first")
        self.assertTrue(all(odf.true_race == odf.preds))

if __name__ == "__main__":
    unittest.main()
