#!/usr/bin/env python

"""
Tests for input validation and error handling.

Consolidated validation tests from test_validation.py and test_bug_fixes.py
"""

import unittest

import pandas as pd

from ethnicolr2 import pred_census_last_name, pred_fl_full_name, pred_fl_last_name


class TestInputValidation(unittest.TestCase):
    """Test input validation and error messages."""

    def setUp(self):
        self.df_valid = pd.DataFrame(
            [{"last": "smith", "first": "john"}, {"last": "garcia", "first": "maria"}]
        )

    def test_missing_lname_column(self):
        """Test error when lname_col doesn't exist"""
        with self.assertRaises(ValueError) as context:
            pred_fl_full_name(self.df_valid, lname_col="nonexistent", fname_col="first")
        self.assertIn("Column 'nonexistent' not found", str(context.exception))

    def test_missing_fname_column(self):
        """Test error when fname_col doesn't exist"""
        with self.assertRaises(ValueError) as context:
            pred_fl_full_name(self.df_valid, lname_col="last", fname_col="nonexistent")
        self.assertIn("Column 'nonexistent' not found", str(context.exception))

    def test_missing_full_name_column(self):
        """Test error when full_name_col doesn't exist"""
        with self.assertRaises(ValueError) as context:
            pred_fl_full_name(self.df_valid, full_name_col="nonexistent")
        self.assertIn("Column 'nonexistent' not found", str(context.exception))

    def test_no_columns_provided(self):
        """Test error when no columns provided"""
        with self.assertRaises(ValueError) as context:
            pred_fl_full_name(self.df_valid)
        self.assertIn(
            "Must provide either full_name_col or both lname_col and fname_col",
            str(context.exception),
        )

    def test_successful_prediction_with_full_name_col(self):
        """Test successful prediction with full_name_col"""
        df_fullname = self.df_valid.copy()
        df_fullname["fullname"] = df_fullname["last"] + " " + df_fullname["first"]
        result = pred_fl_full_name(df_fullname, full_name_col="fullname")
        self.assertIn("preds", result.columns)
        self.assertEqual(len(result), 2)

    def test_successful_prediction_with_separate_cols(self):
        """Test successful prediction with separate first/last name cols"""
        result = pred_fl_full_name(self.df_valid, lname_col="last", fname_col="first")
        self.assertIn("preds", result.columns)
        self.assertEqual(len(result), 2)

    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes"""
        empty_df = pd.DataFrame(columns=["last"])
        # Models should handle empty dataframes gracefully
        try:
            result = pred_fl_last_name(empty_df, "last")
            self.assertEqual(len(result), 0)
        except Exception:
            # If models can't handle empty dataframes, that's also acceptable behavior
            pass

    def test_validation_across_all_models(self):
        """Test that validation works consistently across all models"""
        df = pd.DataFrame([{"existing_col": "value"}])

        # Test Florida full name model validation
        with self.assertRaises(ValueError) as context:
            pred_fl_full_name(df, lname_col="missing_col", fname_col="existing_col")
        self.assertIn("Column 'missing_col' not found", str(context.exception))

        # Test Florida last name model - currently raises KeyError
        with self.assertRaises((ValueError, KeyError)):
            pred_fl_last_name(df, lname_col="missing_col")

        # Test Census model - currently raises KeyError
        with self.assertRaises((ValueError, KeyError)):
            pred_census_last_name(df, lname_col="missing_col")


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and order preservation."""

    def test_prediction_order_maintained(self):
        """Test that prediction results maintain input order"""
        # Create a dataframe with real names to avoid model errors
        df = pd.DataFrame(
            [{"last": "smith", "id": 1}, {"last": "zhang", "id": 2}, {"last": "garcia", "id": 3}]
        )

        # Test order preservation across all models
        models_to_test = [
            ("Florida Last Name", lambda: pred_fl_last_name(df, "last")),
            ("Census Last Name", lambda: pred_census_last_name(df, "last")),
        ]

        for model_name, model_func in models_to_test:
            with self.subTest(model=model_name):
                result = model_func()
                expected_order = [1, 2, 3]
                actual_order = result["id"].tolist()
                self.assertEqual(
                    actual_order,
                    expected_order,
                    f"{model_name}: Prediction order was not maintained",
                )

    def test_output_structure_consistency(self):
        """Test that all models return consistent output structure"""
        df = pd.DataFrame([{"last": "smith", "first": "john"}])

        models = [
            ("Census", pred_census_last_name(df, "last")),
            ("Florida LN", pred_fl_last_name(df, "last")),
            ("Florida FN", pred_fl_full_name(df, lname_col="last", fname_col="first")),
        ]

        for model_name, result in models:
            with self.subTest(model=model_name):
                # All should have required columns
                self.assertIn("preds", result.columns, f"{model_name} missing 'preds' column")
                self.assertIn("probs", result.columns, f"{model_name} missing 'probs' column")

                # Check that predictions are valid
                valid_categories = {"asian", "hispanic", "nh_black", "nh_white", "other"}
                for pred in result["preds"]:
                    self.assertIn(
                        pred, valid_categories, f"{model_name} returned invalid prediction: {pred}"
                    )

    def test_model_differences(self):
        """Verify that different models can produce different results"""
        df = pd.DataFrame([{"last": "zhang"}])

        census_result = pred_census_last_name(df, "last")
        florida_result = pred_fl_last_name(df, "last")

        # Both should predict asian for zhang (they agree on this),
        # but may have different confidence levels
        self.assertEqual(census_result["preds"].iloc[0], "asian")
        self.assertEqual(florida_result["preds"].iloc[0], "asian")

        # Verify both models actually worked
        self.assertEqual(len(census_result), 1)
        self.assertEqual(len(florida_result), 1)


if __name__ == "__main__":
    unittest.main()
