#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests to verify specific bug fixes in v0.1.2
"""

import unittest
import pandas as pd
from ethnicolr2 import pred_census_last_name, pred_fl_last_name, pred_fl_full_name
from ethnicolr2.census_ln import CensusLnData


class TestBugFixes(unittest.TestCase):
    """Test that our v0.1.2 bug fixes are working correctly"""

    def test_census_method_exists_and_works(self):
        """
        Test fix for critical bug: Census model was calling Florida model method
        Verify that pred_census_last_name method exists and works differently from Florida
        """
        df = pd.DataFrame([{"last": "martinez"}])
        
        # Both should work without error
        census_result = pred_census_last_name(df, 'last')
        florida_result = pred_fl_last_name(df, 'last')
        
        # Both should have predictions
        self.assertIn('preds', census_result.columns)
        self.assertIn('preds', florida_result.columns)
        
        # Results should be non-empty
        self.assertEqual(len(census_result), 1)
        self.assertEqual(len(florida_result), 1)
        
        # Both should predict hispanic for martinez, but may have different probabilities
        self.assertIn(census_result['preds'].iloc[0], ['hispanic', 'nh_white', 'nh_black', 'asian', 'other'])
        self.assertIn(florida_result['preds'].iloc[0], ['hispanic', 'nh_white', 'nh_black', 'asian', 'other'])

    def test_census_year_initialization(self):
        """
        Test fix for census_year AttributeError
        Verify that census_year class variable is properly initialized
        """
        # This should not raise AttributeError
        df = pd.DataFrame([{"last": "smith"}])
        
        # Reset class variables to test initialization
        CensusLnData.census_df = None
        CensusLnData.census_year = None
        
        from ethnicolr2.census_ln import census_ln
        
        # This should work without AttributeError
        result_2000 = census_ln(df, 'last', 2000)
        self.assertIn('pctwhite', result_2000.columns)
        
        result_2010 = census_ln(df, 'last', 2010)
        self.assertIn('pcthispanic', result_2010.columns)
        
        # Verify that census_year is properly set
        self.assertEqual(CensusLnData.census_year, 2010)

    def test_prediction_order_preserved(self):
        """
        Test fix for data shuffling during inference
        Verify that prediction order matches input order
        """
        # Create a dataframe with specific order using real names
        df = pd.DataFrame([
            {"last": "smith", "id": 1},
            {"last": "zhang", "id": 2},
            {"last": "garcia", "id": 3}
        ])
        
        result = pred_fl_last_name(df, 'last')
        
        # Verify order is preserved
        expected_ids = [1, 2, 3]
        actual_ids = result['id'].tolist()
        self.assertEqual(actual_ids, expected_ids, "Prediction order not preserved - shuffling may be enabled")

    def test_input_validation_errors(self):
        """
        Test fix for input validation
        Verify that helpful error messages are provided for missing columns
        """
        df = pd.DataFrame([{"existing_col": "value"}])
        
        # Test missing lname_col
        with self.assertRaises(ValueError) as context:
            pred_fl_full_name(df, lname_col="missing_col", fname_col="existing_col")
        self.assertIn("Column 'missing_col' not found", str(context.exception))
        
        # Test missing fname_col
        with self.assertRaises(ValueError) as context:
            pred_fl_full_name(df, lname_col="existing_col", fname_col="missing_col")
        self.assertIn("Column 'missing_col' not found", str(context.exception))
        
        # Test missing full_name_col
        with self.assertRaises(ValueError) as context:
            pred_fl_full_name(df, full_name_col="missing_col")
        self.assertIn("Column 'missing_col' not found", str(context.exception))
        
        # Test no columns provided
        with self.assertRaises(ValueError) as context:
            pred_fl_full_name(df)
        self.assertIn("Must provide either full_name_col or both lname_col and fname_col", str(context.exception))

    def test_model_constants_exist(self):
        """
        Test that model parameter constants are defined
        Verify that magic numbers were replaced with constants
        """
        from ethnicolr2.ethnicolr_class import (
            MAX_NAME_FULLNAME, MAX_NAME_FLORIDA, MAX_NAME_CENSUS,
            HIDDEN_SIZE, BATCH_SIZE, NUM_LAYERS
        )
        
        # Verify constants are defined with expected values
        self.assertEqual(MAX_NAME_FULLNAME, 47)
        self.assertEqual(MAX_NAME_FLORIDA, 30)
        self.assertEqual(MAX_NAME_CENSUS, 15)
        self.assertEqual(HIDDEN_SIZE, 256)
        self.assertEqual(BATCH_SIZE, 64)
        self.assertEqual(NUM_LAYERS, 2)

    def test_resource_loading_compatibility(self):
        """
        Test that resource loading works with both old and new Python versions
        Verify that importlib.resources fallback to pkg_resources works
        """
        # This is tested implicitly by all other tests working
        # If resource loading was broken, model loading would fail
        
        df = pd.DataFrame([{"last": "test"}])
        
        # All these should work without import errors
        census_result = pred_census_last_name(df, 'last')
        florida_result = pred_fl_last_name(df, 'last')
        full_result = pred_fl_full_name(df, lname_col='last', fname_col='last')
        
        self.assertIn('preds', census_result.columns)
        self.assertIn('preds', florida_result.columns)
        self.assertIn('preds', full_result.columns)


class TestRegressionPrevention(unittest.TestCase):
    """Regression tests to prevent future bugs"""

    def test_all_models_produce_valid_predictions(self):
        """Ensure all models produce valid race/ethnicity predictions"""
        df = pd.DataFrame([
            {"last": "smith", "first": "john"},
            {"last": "garcia", "first": "maria"},
            {"last": "zhang", "first": "wei"}
        ])
        
        valid_categories = {'asian', 'hispanic', 'nh_black', 'nh_white', 'other'}
        
        # Test all three models
        census_result = pred_census_last_name(df, 'last')
        florida_ln_result = pred_fl_last_name(df, 'last')
        florida_fn_result = pred_fl_full_name(df, lname_col='last', fname_col='first')
        
        # All predictions should be valid categories
        for result in [census_result, florida_ln_result, florida_fn_result]:
            for pred in result['preds']:
                self.assertIn(pred, valid_categories, f"Invalid prediction: {pred}")

    def test_consistent_column_naming(self):
        """Ensure all models use consistent output column naming"""
        df = pd.DataFrame([{"last": "smith", "first": "john"}])
        
        census_result = pred_census_last_name(df, 'last')
        florida_ln_result = pred_fl_last_name(df, 'last')
        florida_fn_result = pred_fl_full_name(df, lname_col='last', fname_col='first')
        
        # All should have 'preds' column
        for result in [census_result, florida_ln_result, florida_fn_result]:
            self.assertIn('preds', result.columns)
            
        # All should have a probs column (probabilities are now consolidated)
        for result in [census_result, florida_ln_result, florida_fn_result]:
            self.assertIn('probs', result.columns, f"Missing probs column in result: {result.columns}")
            # Verify probs contains actual probability arrays
            self.assertTrue(hasattr(result['probs'].iloc[0], '__len__'), "probs should contain arrays")


if __name__ == "__main__":
    unittest.main()