#!/usr/bin/env python

"""
Tests for all machine learning prediction models.

This module consolidates tests for:
- Florida voter registration models (last name and full name)
- Census-based LSTM models
- Model integration and comparison tests

Consolidated from test_020_pred_census_ln.py, test_040_pred_fl.py, and test_060_pred.py
"""

import unittest

import pandas as pd

from ethnicolr2 import pred_census_last_name, pred_fl_full_name, pred_fl_last_name


class TestFloridaModels(unittest.TestCase):
    """Tests for Florida voter registration models."""

    def setUp(self):
        """Set up test data for Florida models."""
        self.df_fl = pd.DataFrame(
            [
                {"last": "zhang", "first": "simon", "true_race": "asian"},
                {"last": "torres", "first": "raul", "true_race": "hispanic"},
            ]
        )

    def test_florida_last_name_model(self):
        """Test Florida last name only model."""
        result = pred_fl_last_name(self.df_fl, lname_col="last")

        # Verify basic structure
        self.assertIn("preds", result.columns)
        self.assertIn("probs", result.columns)
        self.assertEqual(len(result), 2)

        # Test specific predictions
        zhang_pred = result[result["last"] == "zhang"]["preds"].values[0]
        torres_pred = result[result["last"] == "torres"]["preds"].values[0]
        self.assertEqual(zhang_pred, "asian")
        self.assertEqual(torres_pred, "hispanic")

    def test_florida_full_name_with_last_only(self):
        """Test Florida full name model using only last name."""
        result = pred_fl_full_name(self.df_fl, "last")
        self.assertTrue(all(result.true_race == result.preds))

    def test_florida_full_name_with_both_names(self):
        """Test Florida full name model using both first and last names."""
        result = pred_fl_full_name(self.df_fl, "last", "first")
        self.assertTrue(all(result.true_race == result.preds))

    def test_florida_full_name_with_combined_column(self):
        """Test Florida full name model with a single full name column."""
        df_combined = self.df_fl.copy()
        df_combined["fullname"] = df_combined["last"] + " " + df_combined["first"]

        result = pred_fl_full_name(df_combined, full_name_col="fullname")

        # Verify predictions
        hernandez_pred = result[result["last"] == "torres"]["preds"].values[0]
        zhang_pred = result[result["last"] == "zhang"]["preds"].values[0]
        self.assertEqual(hernandez_pred, "hispanic")
        self.assertEqual(zhang_pred, "asian")


class TestCensusModels(unittest.TestCase):
    """Tests for Census-based LSTM models."""

    def setUp(self):
        """Set up test data for Census models."""
        self.df_census = pd.DataFrame(
            [
                {"last": "smith", "true_race": "nh_white"},
                {"last": "zhang", "true_race": "asian"},
            ]
        )

    def test_census_last_name_predictions(self):
        """Test Census last name LSTM model."""
        result = pred_census_last_name(self.df_census, "last")

        # Verify basic structure
        self.assertIn("preds", result.columns)
        self.assertIn("probs", result.columns)
        self.assertEqual(len(result), 2)

        # Test accuracy - these should match expected races
        self.assertTrue(all(result.true_race == result.preds))


class TestModelIntegration(unittest.TestCase):
    """Integration tests comparing different models and comprehensive functionality."""

    def setUp(self):
        """Set up test data for integration tests."""
        self.df_integration = pd.DataFrame(
            [
                {"last": "hernandez", "first": "hector"},
                {"last": "zhang", "first": "simon"},
            ]
        )

    def test_all_models_produce_predictions(self):
        """Verify all models can make predictions on the same dataset."""
        # Test Florida last name model
        fl_ln_result = pred_fl_last_name(self.df_integration, lname_col="last")
        self.assertEqual(len(fl_ln_result), 2)
        self.assertIn("preds", fl_ln_result.columns)

        # Test Florida full name model
        fl_fn_result = pred_fl_full_name(self.df_integration, lname_col="last", fname_col="first")
        self.assertEqual(len(fl_fn_result), 2)
        self.assertIn("preds", fl_fn_result.columns)

        # Test Census last name model
        census_result = pred_census_last_name(self.df_integration, lname_col="last")
        self.assertEqual(len(census_result), 2)
        self.assertIn("preds", census_result.columns)

    def test_model_prediction_consistency(self):
        """Test that models make expected predictions for known names."""
        # Test Florida last name model
        fl_ln_result = pred_fl_last_name(self.df_integration, lname_col="last")
        hernandez_pred = fl_ln_result[fl_ln_result["last"] == "hernandez"]["preds"].values[0]
        zhang_pred = fl_ln_result[fl_ln_result["last"] == "zhang"]["preds"].values[0]
        self.assertEqual(hernandez_pred, "hispanic")
        self.assertEqual(zhang_pred, "asian")

        # Test Florida full name model
        fl_fn_result = pred_fl_full_name(self.df_integration, lname_col="last", fname_col="first")
        hernandez_pred = fl_fn_result[fl_fn_result["last"] == "hernandez"]["preds"].values[0]
        zhang_pred = fl_fn_result[fl_fn_result["last"] == "zhang"]["preds"].values[0]
        self.assertEqual(hernandez_pred, "hispanic")
        self.assertEqual(zhang_pred, "asian")

        # Test Census model
        census_result = pred_census_last_name(self.df_integration, lname_col="last")
        hernandez_pred = census_result[census_result["last"] == "hernandez"]["preds"].values[0]
        zhang_pred = census_result[census_result["last"] == "zhang"]["preds"].values[0]
        self.assertEqual(hernandez_pred, "hispanic")
        self.assertEqual(zhang_pred, "asian")

    def test_prediction_output_structure(self):
        """Verify all models return properly structured output."""
        models_to_test = [
            ("Florida Last Name", lambda df: pred_fl_last_name(df, lname_col="last")),
            (
                "Florida Full Name",
                lambda df: pred_fl_full_name(df, lname_col="last", fname_col="first"),
            ),
            ("Census Last Name", lambda df: pred_census_last_name(df, lname_col="last")),
        ]

        for model_name, model_func in models_to_test:
            with self.subTest(model=model_name):
                result = model_func(self.df_integration)

                # Check required columns exist
                self.assertIn("preds", result.columns, f"{model_name} missing 'preds' column")
                self.assertIn("probs", result.columns, f"{model_name} missing 'probs' column")

                # Check output size
                self.assertEqual(len(result), 2, f"{model_name} returned wrong number of rows")

                # Check all predictions are valid categories
                valid_categories = {"nh_white", "nh_black", "hispanic", "asian", "other"}
                for pred in result["preds"]:
                    self.assertIn(
                        pred, valid_categories, f"{model_name} returned invalid prediction: {pred}"
                    )


if __name__ == "__main__":
    unittest.main()
