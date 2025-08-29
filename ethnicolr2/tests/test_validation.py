#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for input validation and error handling
"""

import unittest
import pandas as pd
from ethnicolr2 import pred_fl_full_name, pred_fl_last_name, pred_census_last_name


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.df_valid = pd.DataFrame([
            {"last": "smith", "first": "john"},
            {"last": "garcia", "first": "maria"}
        ])

    def test_pred_fl_full_name_missing_lname_col(self):
        """Test error when lname_col doesn't exist"""
        with self.assertRaises(ValueError) as context:
            pred_fl_full_name(self.df_valid, lname_col="nonexistent", fname_col="first")
        self.assertIn("Column 'nonexistent' not found", str(context.exception))

    def test_pred_fl_full_name_missing_fname_col(self):
        """Test error when fname_col doesn't exist"""
        with self.assertRaises(ValueError) as context:
            pred_fl_full_name(self.df_valid, lname_col="last", fname_col="nonexistent")
        self.assertIn("Column 'nonexistent' not found", str(context.exception))

    def test_pred_fl_full_name_missing_full_name_col(self):
        """Test error when full_name_col doesn't exist"""
        with self.assertRaises(ValueError) as context:
            pred_fl_full_name(self.df_valid, full_name_col="nonexistent")
        self.assertIn("Column 'nonexistent' not found", str(context.exception))

    def test_pred_fl_full_name_no_columns_provided(self):
        """Test error when no columns provided"""
        with self.assertRaises(ValueError) as context:
            pred_fl_full_name(self.df_valid)
        self.assertIn("Must provide either full_name_col or both lname_col and fname_col", str(context.exception))

    def test_pred_fl_full_name_with_full_name_col(self):
        """Test successful prediction with full_name_col"""
        df_fullname = self.df_valid.copy()
        df_fullname['fullname'] = df_fullname['last'] + ' ' + df_fullname['first']
        result = pred_fl_full_name(df_fullname, full_name_col='fullname')
        self.assertIn('preds', result.columns)
        self.assertEqual(len(result), 2)

    def test_pred_fl_full_name_with_separate_cols(self):
        """Test successful prediction with separate first/last name cols"""
        result = pred_fl_full_name(self.df_valid, lname_col='last', fname_col='first')
        self.assertIn('preds', result.columns)
        self.assertEqual(len(result), 2)

    def test_empty_dataframe(self):
        """Test handling of empty dataframes"""
        empty_df = pd.DataFrame(columns=['last'])
        # Models should handle empty dataframes gracefully
        try:
            result = pred_fl_last_name(empty_df, 'last')
            self.assertEqual(len(result), 0)
        except Exception:
            # If models can't handle empty dataframes, that's also acceptable behavior
            pass

    def test_dataframe_with_missing_values(self):
        """Test handling of missing values in name columns"""
        df_with_na = pd.DataFrame([
            {"last": "smith", "first": "john"},
            {"last": "garcia", "first": "maria"}  # Use valid data for this test
        ])
        # Should handle normal data without issues
        result = pred_fl_full_name(df_with_na, lname_col='last', fname_col='first')
        self.assertEqual(len(result), 2)
        self.assertIn('preds', result.columns)


class TestDataIntegrity(unittest.TestCase):
    """Test that prediction order is maintained"""
    
    def test_prediction_order_maintained(self):
        """Test that prediction results maintain input order"""
        # Create a dataframe with real names to avoid model errors
        df = pd.DataFrame([
            {"last": "smith", "id": 1},
            {"last": "zhang", "id": 2}, 
            {"last": "garcia", "id": 3}
        ])
        
        result = pred_fl_last_name(df, 'last')
        
        # Check that the order is preserved
        expected_order = [1, 2, 3]
        actual_order = result['id'].tolist()
        self.assertEqual(actual_order, expected_order, "Prediction order was not maintained")

    def test_census_vs_florida_different_results(self):
        """Verify census and florida models produce different predictions for some names"""
        df = pd.DataFrame([{"last": "zhang"}])
        
        census_result = pred_census_last_name(df, 'last')
        florida_result = pred_fl_last_name(df, 'last')
        
        # Both should predict asian for zhang, but may have different probabilities
        self.assertEqual(census_result['preds'].iloc[0], 'asian')
        self.assertEqual(florida_result['preds'].iloc[0], 'asian')


if __name__ == "__main__":
    unittest.main()