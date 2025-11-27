#!/usr/bin/env python

"""
Regression tests to prevent specific bugs that were fixed in previous versions.

This module contains tests that verify specific bug fixes remain working:
- v0.1.2 bug fixes for census model method calling, initialization, and data shuffling
- Resource loading compatibility across Python versions
- Model constant definitions
- Consistent output formatting

Tests in this file should generally not be modified unless the underlying
bug fix changes or new regression issues are discovered.
"""

import unittest

import pandas as pd

from ethnicolr2 import pred_census_last_name, pred_fl_full_name, pred_fl_last_name
from ethnicolr2.census_ln import CensusLnData


class TestSpecificBugFixes(unittest.TestCase):
    """Tests for specific bugs that were fixed in v0.1.2."""

    def test_census_method_exists_and_works_independently(self):
        """
        Regression test for critical bug: Census model was calling Florida model method.

        BUG: pred_census_last_name was incorrectly calling Florida model internally
        FIX: Ensure pred_census_last_name has its own implementation

        This test verifies that census and Florida models work independently.
        """
        df = pd.DataFrame([{"last": "martinez"}])

        # Both models should work without error
        census_result = pred_census_last_name(df, "last")
        florida_result = pred_fl_last_name(df, "last")

        # Verify basic functionality
        self.assertIn("preds", census_result.columns)
        self.assertIn("preds", florida_result.columns)
        self.assertEqual(len(census_result), 1)
        self.assertEqual(len(florida_result), 1)

        # Verify predictions are valid categories
        valid_categories = ["hispanic", "nh_white", "nh_black", "asian", "other"]
        self.assertIn(census_result["preds"].iloc[0], valid_categories)
        self.assertIn(florida_result["preds"].iloc[0], valid_categories)

    def test_census_year_initialization_fix(self):
        """
        Regression test for census_year AttributeError.

        BUG: CensusLnData.census_year was not properly initialized, causing AttributeError
        FIX: Proper initialization of class variables

        This test verifies census_year is properly handled during initialization.
        """
        df = pd.DataFrame([{"last": "smith"}])

        # Reset class variables to simulate fresh import
        original_df = CensusLnData.census_df
        original_year = CensusLnData.census_year

        try:
            CensusLnData.census_df = None
            CensusLnData.census_year = None

            from ethnicolr2.census_ln import census_ln

            # These operations should not raise AttributeError
            result_2000 = census_ln(df, "last", 2000)
            self.assertIn("pctwhite", result_2000.columns)

            result_2010 = census_ln(df, "last", 2010)
            self.assertIn("pcthispanic", result_2010.columns)

            # Verify census_year is properly set after operations
            self.assertEqual(CensusLnData.census_year, 2010)

        finally:
            # Restore original state
            CensusLnData.census_df = original_df
            CensusLnData.census_year = original_year

    def test_prediction_order_preservation_fix(self):
        """
        Regression test for data shuffling during inference.

        BUG: DataLoader was shuffling data during inference, breaking input/output order
        FIX: Disabled shuffling for inference DataLoader

        This test verifies prediction order matches input order.
        """
        df = pd.DataFrame(
            [
                {"last": "smith", "id": 1},
                {"last": "zhang", "id": 2},
                {"last": "garcia", "id": 3},
            ]
        )

        # Test across all models
        models_to_test = [
            ("Florida Last Name", lambda: pred_fl_last_name(df, "last")),
            ("Census Last Name", lambda: pred_census_last_name(df, "last")),
            (
                "Florida Full Name",
                lambda: pred_fl_full_name(df, lname_col="last", fname_col="last"),
            ),
        ]

        for model_name, model_func in models_to_test:
            with self.subTest(model=model_name):
                result = model_func()

                # Verify order is preserved
                expected_ids = [1, 2, 3]
                actual_ids = result["id"].tolist()
                self.assertEqual(
                    actual_ids,
                    expected_ids,
                    f"{model_name}: Data shuffling detected - order not preserved",
                )

    def test_model_constants_defined(self):
        """
        Regression test for magic number replacement.

        BUG: Magic numbers scattered throughout code made maintenance difficult
        FIX: Centralized constants in ethnicolr_class module

        This test verifies that model constants are properly defined.
        """
        try:
            from ethnicolr2.ethnicolr_class import (
                BATCH_SIZE,
                HIDDEN_SIZE,
                MAX_NAME_CENSUS,
                MAX_NAME_FLORIDA,
                MAX_NAME_FULLNAME,
                NUM_LAYERS,
            )

            # Verify constants have expected values
            self.assertEqual(MAX_NAME_FULLNAME, 47)
            self.assertEqual(MAX_NAME_FLORIDA, 30)
            self.assertEqual(MAX_NAME_CENSUS, 15)
            self.assertEqual(HIDDEN_SIZE, 256)
            self.assertEqual(BATCH_SIZE, 64)
            self.assertEqual(NUM_LAYERS, 2)

        except ImportError as e:
            self.fail(f"Model constants not properly defined: {e}")

    def test_resource_loading_compatibility(self):
        """
        Regression test for Python version compatibility.

        BUG: importlib.resources not available in Python < 3.9
        FIX: Graceful fallback to pkg_resources for older Python versions

        This test verifies resource loading works across Python versions.
        """
        df = pd.DataFrame([{"last": "test"}])

        # If resource loading was broken, these would fail with import errors
        try:
            census_result = pred_census_last_name(df, "last")
            florida_ln_result = pred_fl_last_name(df, "last")
            florida_fn_result = pred_fl_full_name(
                df, lname_col="last", fname_col="last"
            )

            # Verify all models loaded and worked
            for result in [census_result, florida_ln_result, florida_fn_result]:
                self.assertIn("preds", result.columns)

        except ImportError as e:
            self.fail(f"Resource loading compatibility issue: {e}")


class TestRegressionPrevention(unittest.TestCase):
    """Tests to prevent future regressions in core functionality."""

    def test_all_models_produce_valid_predictions(self):
        """Ensure all models consistently produce valid race/ethnicity predictions."""
        df = pd.DataFrame(
            [
                {"last": "smith", "first": "john"},
                {"last": "garcia", "first": "maria"},
                {"last": "zhang", "first": "wei"},
            ]
        )

        valid_categories = {"asian", "hispanic", "nh_black", "nh_white", "other"}

        # Test all models
        models = [
            ("Census", pred_census_last_name(df, "last")),
            ("Florida LN", pred_fl_last_name(df, "last")),
            ("Florida FN", pred_fl_full_name(df, lname_col="last", fname_col="first")),
        ]

        for model_name, result in models:
            with self.subTest(model=model_name):
                # All predictions should be valid categories
                for pred in result["preds"]:
                    self.assertIn(
                        pred,
                        valid_categories,
                        f"{model_name} returned invalid prediction: {pred}",
                    )

    def test_consistent_column_naming_across_models(self):
        """Ensure all models use consistent output column naming."""
        df = pd.DataFrame([{"last": "smith", "first": "john"}])

        models = [
            ("Census", pred_census_last_name(df, "last")),
            ("Florida LN", pred_fl_last_name(df, "last")),
            ("Florida FN", pred_fl_full_name(df, lname_col="last", fname_col="first")),
        ]

        for model_name, result in models:
            with self.subTest(model=model_name):
                # All models should have consistent column names
                self.assertIn(
                    "preds", result.columns, f"{model_name} missing 'preds' column"
                )
                self.assertIn(
                    "probs", result.columns, f"{model_name} missing 'probs' column"
                )

                # Verify probs column contains proper probability data
                self.assertTrue(
                    hasattr(result["probs"].iloc[0], "__len__"),
                    f"{model_name} probs column should contain arrays/dicts",
                )

    def test_model_output_structure_stability(self):
        """Verify that model output structure remains stable across versions."""
        df = pd.DataFrame([{"last": "test", "first": "name"}])

        models_to_test = [
            ("Census", lambda: pred_census_last_name(df, "last")),
            ("Florida LN", lambda: pred_fl_last_name(df, "last")),
            (
                "Florida FN",
                lambda: pred_fl_full_name(df, lname_col="last", fname_col="first"),
            ),
        ]

        for model_name, model_func in models_to_test:
            with self.subTest(model=model_name):
                result = model_func()

                # Verify basic structure requirements
                self.assertIsInstance(
                    result, pd.DataFrame, f"{model_name} should return DataFrame"
                )
                self.assertEqual(
                    len(result),
                    1,
                    f"{model_name} should return same number of rows as input",
                )

                # Verify required columns exist
                required_columns = ["preds", "probs"]
                for col in required_columns:
                    self.assertIn(
                        col,
                        result.columns,
                        f"{model_name} missing required column: {col}",
                    )


if __name__ == "__main__":
    unittest.main()
