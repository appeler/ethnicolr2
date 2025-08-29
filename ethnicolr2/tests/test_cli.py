#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for command-line interface functionality
"""

import unittest
import tempfile
import os
import csv
from unittest.mock import patch
from ethnicolr2.pred_fl_fn_lstm import main as fl_fn_main
from ethnicolr2.pred_fl_ln_lstm import main as fl_ln_main
from ethnicolr2.pred_cen_ln_lstm import main as cen_ln_main
from ethnicolr2.census_ln import main as census_main


class TestCLI(unittest.TestCase):
    def setUp(self):
        # Create temporary test file
        self.temp_dir = tempfile.mkdtemp()
        self.input_file = os.path.join(self.temp_dir, "test_input.csv")
        
        # Write test data
        with open(self.input_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['first', 'last'])
            writer.writerow(['john', 'smith'])
            writer.writerow(['maria', 'garcia'])

    def tearDown(self):
        # Clean up temp files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_fl_full_name_cli_with_separate_columns(self):
        """Test Florida full name CLI with separate first/last columns"""
        output_file = os.path.join(self.temp_dir, "output.csv")
        
        args = [
            self.input_file,
            "--output", output_file,
            "--last", "last", 
            "--first", "first"
        ]
        
        # Should not raise exception
        try:
            fl_fn_main(args)
        except SystemExit:
            pass  # main() calls sys.exit(), which is expected
        
        # Check output file was created
        self.assertTrue(os.path.exists(output_file))

    def test_fl_last_name_cli(self):
        """Test Florida last name CLI"""
        output_file = os.path.join(self.temp_dir, "output.csv")
        
        args = [
            self.input_file,
            "--output", output_file,
            "--last", "last"
        ]
        
        try:
            fl_ln_main(args)
        except SystemExit:
            pass  # main() calls sys.exit(), which is expected
            
        self.assertTrue(os.path.exists(output_file))

    def test_census_last_name_cli(self):
        """Test Census last name CLI"""
        output_file = os.path.join(self.temp_dir, "output.csv")
        
        args = [
            self.input_file,
            "--output", output_file,
            "--last", "last"
        ]
        
        try:
            cen_ln_main(args)
        except SystemExit:
            pass  # main() calls sys.exit(), which is expected
            
        self.assertTrue(os.path.exists(output_file))

    def test_census_lookup_cli_2000(self):
        """Test Census lookup CLI for year 2000"""
        output_file = os.path.join(self.temp_dir, "output.csv")
        
        args = [
            self.input_file,
            "--output", output_file,
            "--last", "last",
            "--year", "2000"
        ]
        
        try:
            census_main(args)
        except SystemExit:
            pass  # main() calls sys.exit(), which is expected
            
        self.assertTrue(os.path.exists(output_file))

    def test_census_lookup_cli_2010(self):
        """Test Census lookup CLI for year 2010"""
        output_file = os.path.join(self.temp_dir, "output.csv")
        
        args = [
            self.input_file,
            "--output", output_file,
            "--last", "last",
            "--year", "2010"
        ]
        
        try:
            census_main(args)
        except SystemExit:
            pass  # main() calls sys.exit(), which is expected
            
        self.assertTrue(os.path.exists(output_file))

    @patch('builtins.print')
    def test_arg_parser_prints_args(self, mock_print):
        """Test that arg_parser prints arguments (for coverage of utils.py)"""
        from ethnicolr2.utils import arg_parser
        
        args = [
            "dummy_file.csv",
            "--output", "output.csv", 
            "--last", "last_col",
            "--year", "2010"
        ]
        
        parsed_args = arg_parser(
            args,
            title="Test parser",
            default_out="default.csv",
            default_year=2000,
            year_choices=[2000, 2010]
        )
        
        # Should have printed the parsed arguments
        mock_print.assert_called()
        self.assertEqual(parsed_args.input, "dummy_file.csv")
        self.assertEqual(parsed_args.year, 2010)


if __name__ == "__main__":
    unittest.main()