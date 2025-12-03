#!/usr/bin/env python

"""
Tests for input data edge cases and boundary conditions.

This module tests various problematic inputs that can occur in real-world usage:
- Malformed CSV data
- Unicode and encoding issues
- Boundary value conditions
- Large datasets
- Special characters and edge cases
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import ethnicolr2


class TestCSVParsingEdgeCases(unittest.TestCase):
    """Test CSV parsing and data loading edge cases."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_empty_csv_file(self):
        """Test handling of completely empty CSV files."""
        empty_file = Path(self.temp_dir) / "empty.csv"
        empty_file.write_text("")

        # Should raise an error or handle gracefully
        with self.assertRaises((pd.errors.EmptyDataError, ValueError)):
            df = pd.read_csv(empty_file)
            ethnicolr2.pred_fl_last_name(df, "last")

    def test_csv_with_only_header(self):
        """Test CSV with header but no data rows."""
        header_only_file = Path(self.temp_dir) / "header_only.csv"
        header_only_file.write_text("last,first\n")

        df = pd.read_csv(header_only_file)
        # Should handle empty DataFrame gracefully
        try:
            result = ethnicolr2.pred_fl_last_name(df, "last")
            self.assertEqual(len(result), 0)
        except ValueError:
            # Also acceptable to raise an error for empty data
            pass

    def test_csv_with_malformed_rows(self):
        """Test CSV with inconsistent number of columns."""
        malformed_file = Path(self.temp_dir) / "malformed.csv"
        malformed_content = """last,first
smith,john
garcia,maria,extra_column
jones"""
        malformed_file.write_text(malformed_content)

        # pandas should handle this, but let's test our functions are robust
        try:
            df = pd.read_csv(malformed_file)
            # Should still work if last column exists
            if "last" in df.columns:
                result = ethnicolr2.pred_fl_last_name(df, "last")
                self.assertGreater(len(result), 0)
        except pd.errors.ParserError:
            # Also acceptable for malformed CSV to fail at pandas level
            pass

    def test_csv_with_quoted_fields(self):
        """Test CSV with quoted fields containing commas."""
        quoted_file = Path(self.temp_dir) / "quoted.csv"
        quoted_content = '''last,first,notes
"Smith, Jr.",John,"Has a comma, in name"
Garcia,Maria,"Normal entry"'''
        quoted_file.write_text(quoted_content)

        df = pd.read_csv(quoted_file)
        result = ethnicolr2.pred_fl_last_name(df, "last")

        # Should handle quoted fields correctly
        self.assertEqual(len(result), 2)
        self.assertIn("Smith, Jr.", result["last"].values)

    def test_different_csv_dialects(self):
        """Test different CSV dialects (semicolon, tab-separated)."""
        # Semicolon-separated
        semicolon_file = Path(self.temp_dir) / "semicolon.csv"
        semicolon_file.write_text("last;first\nsmith;john\ngarcia;maria")

        df = pd.read_csv(semicolon_file, sep=";")
        result = ethnicolr2.pred_fl_last_name(df, "last")
        self.assertEqual(len(result), 2)

        # Tab-separated
        tab_file = Path(self.temp_dir) / "tab.tsv"
        tab_file.write_text("last\tfirst\nsmith\tjohn\ngarcia\tmaria")

        df = pd.read_csv(tab_file, sep="\t")
        result = ethnicolr2.pred_fl_last_name(df, "last")
        self.assertEqual(len(result), 2)


class TestUnicodeAndEncodingIssues(unittest.TestCase):
    """Test Unicode handling and encoding edge cases."""

    def test_unicode_names(self):
        """Test names with Unicode characters."""
        unicode_df = pd.DataFrame(
            {
                "last": ["García", "Müller", "Žáček", "Иванов", "张", "أحمد"],
                "first": ["José", "Hans", "Pavel", "Иван", "伟", "محمد"],
            }
        )

        # Should handle Unicode names gracefully
        result = ethnicolr2.pred_fl_last_name(unicode_df, "last")
        self.assertEqual(len(result), 6)
        self.assertIn("preds", result.columns)

        # Test full name model with Unicode
        result_full = ethnicolr2.pred_fl_full_name(
            unicode_df, lname_col="last", fname_col="first"
        )
        self.assertEqual(len(result_full), 6)

    def test_mixed_encoding_issues(self):
        """Test handling of mixed encoding issues."""
        # Create DataFrame with mixed content
        mixed_df = pd.DataFrame(
            {
                "last": ["smith", "garcía", "müller"],  # Mixed ASCII and Unicode
                "first": ["john", "josé", "hans"],
            }
        )

        result = ethnicolr2.pred_fl_last_name(mixed_df, "last")
        self.assertEqual(len(result), 3)

    def test_emoji_and_special_characters(self):
        """Test names with emojis and other special characters."""
        special_df = pd.DataFrame(
            {
                "last": [
                    "Smith 👨",
                    "O'Connor",
                    "D'Angelo",
                    "Smith-Jones",
                    "Van Der Berg",
                ],
                "first": ["John", "Patrick", "Maria", "Anne", "Hans"],
            }
        )

        # Should handle special characters gracefully
        result = ethnicolr2.pred_fl_last_name(special_df, "last")
        self.assertEqual(len(result), 5)

    def test_extremely_long_unicode_names(self):
        """Test very long Unicode names."""
        very_long_name = "García" * 50  # 300 characters
        long_df = pd.DataFrame({"last": [very_long_name, "Smith"]})

        # Should handle long names (truncation should work)
        result = ethnicolr2.pred_fl_last_name(long_df, "last")
        self.assertEqual(len(result), 2)


class TestBoundaryValueConditions(unittest.TestCase):
    """Test boundary conditions and edge values."""

    def test_single_character_names(self):
        """Test single character names."""
        single_char_df = pd.DataFrame(
            {
                "last": ["A", "B", "李"],  # ASCII and Unicode single chars
                "first": ["X", "Y", "Z"],
            }
        )

        result = ethnicolr2.pred_fl_last_name(single_char_df, "last")
        self.assertEqual(len(result), 3)

    def test_empty_string_names(self):
        """Test empty string names."""
        empty_df = pd.DataFrame(
            {"last": ["", "Smith", ""], "first": ["John", "", "Maria"]}
        )

        # Should handle empty strings gracefully
        # Models might predict something or filter them out
        try:
            result = ethnicolr2.pred_fl_last_name(empty_df, "last")
            # If it works, verify structure is correct
            self.assertIn("preds", result.columns)
        except (ValueError, KeyError):
            # Also acceptable to fail on empty names
            pass

    def test_whitespace_only_names(self):
        """Test names that are only whitespace."""
        whitespace_df = pd.DataFrame(
            {
                "last": ["   ", "\t", "\n", "Smith"],
                "first": ["John", "   ", "Maria", "Anne"],
            }
        )

        # Test both models
        try:
            result = ethnicolr2.pred_fl_last_name(whitespace_df, "last")
            self.assertIn("preds", result.columns)
        except (ValueError, KeyError):
            # Acceptable to fail on whitespace-only names
            pass

    def test_numeric_names(self):
        """Test names that are numeric."""
        numeric_df = pd.DataFrame(
            {"last": ["123", "Smith", "42"], "first": ["John", "456", "Maria"]}
        )

        # Should handle numeric strings
        result = ethnicolr2.pred_fl_last_name(numeric_df, "last")
        self.assertEqual(len(result), 3)

    def test_names_with_only_punctuation(self):
        """Test names with only punctuation marks."""
        punct_df = pd.DataFrame(
            {"last": ["!!!", "Smith", "@#$"], "first": ["John", "...", "Maria"]}
        )

        result = ethnicolr2.pred_fl_last_name(punct_df, "last")
        self.assertEqual(len(result), 3)


class TestDataTypeIssues(unittest.TestCase):
    """Test data type conversion and mixed type issues."""

    def test_mixed_data_types_in_column(self):
        """Test columns with mixed data types."""
        mixed_df = pd.DataFrame(
            {
                "last": ["Smith", 123, None, "Garcia", 45.67],
                "first": ["John", "Maria", "Carlos", None, "Anne"],
            }
        )

        # Convert to string as pandas would
        mixed_df["last"] = mixed_df["last"].astype(str)
        mixed_df["first"] = mixed_df["first"].astype(str)

        # Should handle mixed types that become strings
        try:
            result = ethnicolr2.pred_fl_last_name(mixed_df, "last")
            self.assertIn("preds", result.columns)
        except (ValueError, KeyError):
            # Acceptable if model can't handle converted types
            pass

    def test_float_values_in_name_columns(self):
        """Test numeric float values in name columns."""
        float_df = pd.DataFrame(
            {"last": [123.456, "Smith", np.nan], "first": ["John", 789.012, "Maria"]}
        )

        # Let pandas convert to string
        float_df = float_df.astype(str)

        result = ethnicolr2.pred_fl_last_name(float_df, "last")
        self.assertIn("preds", result.columns)

    def test_boolean_values_in_name_columns(self):
        """Test boolean values in name columns."""
        bool_df = pd.DataFrame(
            {"last": [True, "Smith", False], "first": ["John", False, "Maria"]}
        )

        bool_df = bool_df.astype(str)

        result = ethnicolr2.pred_fl_last_name(bool_df, "last")
        self.assertEqual(len(result), 3)


class TestLargeDatasetHandling(unittest.TestCase):
    """Test handling of large datasets."""

    def test_moderate_size_dataset(self):
        """Test moderately sized dataset (1000 rows)."""
        # Generate test data
        names = ["Smith", "Garcia", "Johnson", "Zhang", "Patel"] * 200
        large_df = pd.DataFrame({"last": names, "first": ["John"] * len(names)})

        result = ethnicolr2.pred_fl_last_name(large_df, "last")
        self.assertEqual(len(result), 1000)
        self.assertIn("preds", result.columns)
        self.assertIn("probs", result.columns)

    def test_dataset_with_many_unique_names(self):
        """Test dataset with many unique names."""
        # Generate unique names
        unique_names = [f"Name{i}" for i in range(500)]
        unique_df = pd.DataFrame(
            {"last": unique_names, "first": ["John"] * len(unique_names)}
        )

        result = ethnicolr2.pred_fl_last_name(unique_df, "last")
        self.assertEqual(len(result), 500)

    def test_dataset_with_many_duplicate_names(self):
        """Test dataset with many duplicate names (tests caching efficiency)."""
        # Many duplicates - should benefit from caching
        duplicate_df = pd.DataFrame(
            {"last": ["Smith"] * 1000, "first": ["John"] * 1000}
        )

        result = ethnicolr2.pred_fl_last_name(duplicate_df, "last")
        self.assertEqual(len(result), 1000)
        # All predictions should be the same
        unique_preds = result["preds"].nunique()
        self.assertEqual(unique_preds, 1)


class TestSpecialCharacterHandling(unittest.TestCase):
    """Test handling of various special characters."""

    def test_names_with_apostrophes(self):
        """Test names with apostrophes and contractions."""
        apostrophe_df = pd.DataFrame(
            {
                "last": ["O'Connor", "D'Angelo", "McDonald's", "L'Oreal"],
                "first": ["Patrick", "Maria", "Ronald", "Anne"],
            }
        )

        result = ethnicolr2.pred_fl_last_name(apostrophe_df, "last")
        self.assertEqual(len(result), 4)

    def test_names_with_hyphens(self):
        """Test hyphenated names."""
        hyphen_df = pd.DataFrame(
            {
                "last": ["Smith-Jones", "García-López", "Van-Der-Berg", "Mary-Anne"],
                "first": ["John", "Carlos", "Hans", "Sue"],
            }
        )

        result = ethnicolr2.pred_fl_last_name(hyphen_df, "last")
        self.assertEqual(len(result), 4)

    def test_names_with_spaces(self):
        """Test names with internal spaces."""
        space_df = pd.DataFrame(
            {
                "last": ["Van Der Berg", "De La Cruz", "Al Rashid", "Mac Donald"],
                "first": ["Hans", "Carlos", "Ahmed", "Ronald"],
            }
        )

        result = ethnicolr2.pred_fl_last_name(space_df, "last")
        self.assertEqual(len(result), 4)

    def test_names_with_numbers(self):
        """Test names that include numbers."""
        number_df = pd.DataFrame(
            {
                "last": ["Smith Jr.", "John 3rd", "Louis XIV", "Agent 007"],
                "first": ["John", "Robert", "King", "James"],
            }
        )

        result = ethnicolr2.pred_fl_last_name(number_df, "last")
        self.assertEqual(len(result), 4)


class TestColumnHandlingEdgeCases(unittest.TestCase):
    """Test edge cases in column handling and validation."""

    def test_column_name_variations(self):
        """Test various column name formats."""
        # Column names with spaces, special chars
        df = pd.DataFrame(
            {
                "Last Name": ["Smith", "Garcia"],
                "First Name": ["John", "Maria"],
                "last_name": ["Johnson", "Zhang"],
                "first_name": ["Bob", "Wei"],
            }
        )

        # Should work with column names containing spaces
        result = ethnicolr2.pred_fl_last_name(df, "Last Name")
        self.assertEqual(len(result), 2)

        # Should work with underscore column names
        result2 = ethnicolr2.pred_fl_last_name(df, "last_name")
        self.assertEqual(len(result2), 2)

    def test_case_sensitive_column_names(self):
        """Test case sensitivity in column names."""
        df = pd.DataFrame(
            {
                "LAST": ["Smith", "Garcia"],
                "last": ["Johnson", "Zhang"],
                "Last": ["Wilson", "Patel"],
            }
        )

        # Should be case-sensitive
        result = ethnicolr2.pred_fl_last_name(df, "LAST")
        self.assertEqual(len(result), 2)

        result2 = ethnicolr2.pred_fl_last_name(df, "last")
        self.assertEqual(len(result2), 2)

    def test_duplicate_column_names(self):
        """Test DataFrames with duplicate column names."""
        # This creates a DataFrame where pandas handles duplicate names
        df = pd.DataFrame([["Smith", "Garcia", "Johnson"], ["John", "Maria", "Bob"]]).T
        df.columns = ["last", "last", "first"]  # Duplicate 'last' columns

        # Should handle duplicate columns (pandas auto-renames)
        try:
            # This might work or fail depending on pandas behavior
            result = ethnicolr2.pred_fl_last_name(df, "last")
            self.assertIn("preds", result.columns)
        except (ValueError, KeyError):
            # Acceptable to fail with duplicate columns
            pass


if __name__ == "__main__":
    unittest.main()
