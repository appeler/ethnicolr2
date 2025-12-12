#!/usr/bin/env python

"""
Tests for CLI error handling and file system edge cases.

This module tests various CLI failure scenarios:
- File system permission errors
- Invalid file paths and arguments
- Disk space issues
- Process interruption handling
- Invalid CSV formats and encodings
"""

import csv
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from ethnicolr2.census_ln import main as census_main
from ethnicolr2.pred_cen_ln_lstm import main as cen_ln_main
from ethnicolr2.pred_fl_fn_lstm import main as fl_fn_main
from ethnicolr2.pred_fl_ln_lstm import main as fl_ln_main


class TestCLIFileSystemErrors(unittest.TestCase):
    """Test CLI handling of file system errors."""

    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create valid test input file
        self.input_file = self.temp_dir / "test_input.csv"
        with open(self.input_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["first", "last"])
            writer.writerow(["john", "smith"])
            writer.writerow(["maria", "garcia"])

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_nonexistent_input_file(self):
        """Test behavior with non-existent input file."""
        nonexistent_file = "/path/that/does/not/exist.csv"
        output_file = self.temp_dir / "output.csv"

        # Test Florida last name CLI
        result = self.runner.invoke(
            fl_ln_main,
            [nonexistent_file, "--output", output_file, "--last-name-col", "last"],
        )

        # Should fail gracefully
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("does not exist", result.output)

    def test_permission_denied_input_file(self):
        """Test behavior when input file permissions are denied."""
        # Create file and remove read permissions
        restricted_file = self.temp_dir / "restricted.csv"
        Path(restricted_file).write_text("last,first\nsmith,john")

        # Remove read permissions (on Unix systems)
        try:
            restricted_file.chmod(0o000)

            result = self.runner.invoke(
                fl_ln_main,
                [
                    restricted_file,
                    "--output",
                    self.temp_dir / "output.csv",
                    "--last-name-col",
                    "last",
                ],
            )

            # Should fail gracefully
            self.assertNotEqual(result.exit_code, 0)

        except (OSError, PermissionError):
            # Skip test if we can't change permissions (e.g., on Windows)
            self.skipTest("Cannot test permission denied on this system")
        finally:
            # Restore permissions for cleanup
            try:
                restricted_file.chmod(0o644)
            except OSError:
                pass

    def test_permission_denied_output_directory(self):
        """Test behavior when output directory permissions are denied."""
        # Create directory and remove write permissions
        restricted_dir = self.temp_dir / "restricted_dir"
        restricted_dir.mkdir(parents=True, exist_ok=True)

        try:
            restricted_dir.chmod(0o444)  # Read-only

            output_file = restricted_dir / "output.csv"

            result = self.runner.invoke(
                fl_ln_main,
                [self.input_file, "--output", output_file, "--last-name-col", "last"],
            )

            # Should fail gracefully
            self.assertNotEqual(result.exit_code, 0)

        except (OSError, PermissionError):
            self.skipTest("Cannot test permission denied on this system")
        finally:
            try:
                restricted_dir.chmod(0o755)
            except OSError:
                pass

    def test_output_to_directory_instead_of_file(self):
        """Test specifying a directory as output instead of file."""
        output_dir = self.temp_dir / "output_dir"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.runner.invoke(
            fl_ln_main,
            [
                self.input_file,
                "--output",
                output_dir,  # Directory, not file
                "--last-name-col",
                "last",
            ],
        )

        # Should fail or handle gracefully
        # The exact behavior depends on implementation
        if result.exit_code != 0:
            self.assertIn("Error", result.output)

    def test_nonexistent_output_directory(self):
        """Test output path with non-existent parent directory."""
        nonexistent_dir = "/path/that/does/not/exist"
        output_file = Path(nonexistent_dir) / "output.csv"

        result = self.runner.invoke(
            fl_ln_main,
            [self.input_file, "--output", output_file, "--last-name-col", "last"],
        )

        # Should fail gracefully
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("does not exist", result.output)


class TestCLIInvalidArguments(unittest.TestCase):
    """Test CLI handling of invalid arguments."""

    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create valid test input file
        self.input_file = self.temp_dir / "test_input.csv"
        with open(self.input_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["first", "last"])
            writer.writerow(["john", "smith"])

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_missing_required_column_argument(self):
        """Test missing required column arguments."""
        output_file = self.temp_dir / "output.csv"

        # Missing last-name-col argument
        result = self.runner.invoke(
            fl_ln_main,
            [
                self.input_file,
                "--output",
                output_file,
                # Missing --last-name-col
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Missing option", result.output)

    def test_invalid_column_name(self):
        """Test specifying column name that doesn't exist."""
        output_file = self.temp_dir / "output.csv"

        result = self.runner.invoke(
            fl_ln_main,
            [
                self.input_file,
                "--output",
                output_file,
                "--last-name-col",
                "nonexistent_column",
            ],
        )

        # Should fail when trying to process
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Error", result.output)

    def test_invalid_year_argument(self):
        """Test invalid year argument for census functions."""
        output_file = self.temp_dir / "output.csv"

        # Test invalid year
        result = self.runner.invoke(
            census_main,
            [
                self.input_file,
                "--output",
                output_file,
                "--last-name-col",
                "last",
                "--year",
                "1995",  # Invalid year
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        # Should mention invalid choice
        self.assertIn("Invalid value", result.output)

    def test_conflicting_arguments_full_name(self):
        """Test conflicting arguments for full name functions."""
        # This tests the validation logic in full name functions
        output_file = self.temp_dir / "output.csv"

        # Neither full name col nor separate first/last cols provided
        result = self.runner.invoke(
            fl_fn_main,
            [
                self.input_file,
                "--output",
                output_file,
                # No column specifications
            ],
        )

        # Should fail due to missing required arguments
        self.assertNotEqual(result.exit_code, 0)

    def test_help_command_works(self):
        """Test that help commands work properly."""
        # Test help for each CLI
        for cli_func in [fl_ln_main, fl_fn_main, cen_ln_main, census_main]:
            result = self.runner.invoke(cli_func, ["--help"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Usage:", result.output)


class TestCLIMalformedDataHandling(unittest.TestCase):
    """Test CLI handling of malformed CSV data."""

    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_empty_csv_file(self):
        """Test completely empty CSV file."""
        empty_file = self.temp_dir / "empty.csv"
        Path(empty_file).write_text("")

        output_file = self.temp_dir / "output.csv"

        result = self.runner.invoke(
            fl_ln_main, [empty_file, "--output", output_file, "--last-name-col", "last"]
        )

        # Should handle empty file gracefully
        self.assertNotEqual(result.exit_code, 0)

    def test_csv_with_only_header(self):
        """Test CSV with header but no data."""
        header_only_file = self.temp_dir / "header_only.csv"
        Path(header_only_file).write_text("last,first\n")

        output_file = self.temp_dir / "output.csv"

        result = self.runner.invoke(
            fl_ln_main,
            [header_only_file, "--output", output_file, "--last-name-col", "last"],
        )

        # Should handle gracefully (might succeed with empty output)
        if result.exit_code == 0:
            # If it succeeds, output file should exist (even if empty)
            self.assertTrue(output_file.exists())

    def test_csv_with_malformed_encoding(self):
        """Test CSV with encoding issues."""
        # Create file with mixed encoding
        malformed_file = self.temp_dir / "malformed.csv"
        with open(malformed_file, "wb") as f:
            # Write header in UTF-8
            f.write(b"last,first\n")
            # Write problematic content
            f.write("García,José\n".encode())
            f.write("Müller,Hans\n".encode("latin-1"))  # Different encoding

        output_file = self.temp_dir / "output.csv"

        result = self.runner.invoke(
            fl_ln_main,
            [malformed_file, "--output", output_file, "--last-name-col", "last"],
        )

        # May succeed or fail depending on pandas handling
        # Either outcome is acceptable as long as it's graceful
        if result.exit_code != 0:
            self.assertIn("Error", result.output)

    def test_csv_with_inconsistent_columns(self):
        """Test CSV with rows having different number of columns."""
        inconsistent_file = self.temp_dir / "inconsistent.csv"
        with open(inconsistent_file, "w") as f:
            f.write("last,first\n")
            f.write("Smith,John\n")
            f.write("Garcia,Maria,Extra\n")  # Extra column
            f.write("Johnson\n")  # Missing column

        output_file = self.temp_dir / "output.csv"

        result = self.runner.invoke(
            fl_ln_main,
            [inconsistent_file, "--output", output_file, "--last-name-col", "last"],
        )

        # pandas usually handles this, so it might succeed
        if result.exit_code == 0:
            self.assertTrue(output_file.exists())


class TestCLIResourceConstraints(unittest.TestCase):
    """Test CLI behavior under resource constraints."""

    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_extremely_large_input_file(self):
        """Test handling of very large input files."""
        # Create a moderately large CSV file
        large_file = self.temp_dir / "large.csv"
        with open(large_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["last", "first"])
            # Write 5000 rows
            for i in range(5000):
                writer.writerow([f"LastName{i}", f"FirstName{i}"])

        output_file = self.temp_dir / "output.csv"

        # Should handle large files (might take time but shouldn't crash)
        result = self.runner.invoke(
            fl_ln_main,
            [large_file, "--output", output_file, "--last-name-col", "last"],
            catch_exceptions=True,
        )

        # Either succeeds or fails gracefully with memory error
        if result.exit_code == 0:
            self.assertTrue(output_file.exists())
            # Output should have same number of rows plus header
            with open(output_file) as f:
                lines = sum(1 for line in f)
                self.assertGreater(lines, 5000)  # Header + data rows

    @patch("pandas.DataFrame.to_csv")
    def test_disk_full_during_output(self, mock_to_csv):
        """Test behavior when disk becomes full during output."""
        # Mock to_csv to raise OSError (disk full)
        mock_to_csv.side_effect = OSError("No space left on device")

        input_file = self.temp_dir / "test.csv"
        Path(input_file).write_text("last,first\nSmith,John")

        output_file = self.temp_dir / "output.csv"

        result = self.runner.invoke(
            fl_ln_main, [input_file, "--output", output_file, "--last-name-col", "last"]
        )

        # Should fail gracefully
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Error", result.output)

    def test_interrupted_processing(self):
        """Test handling of process interruption."""
        # This is difficult to test reliably, but we can test
        # that error handling exists for KeyboardInterrupt

        input_file = self.temp_dir / "test.csv"
        Path(input_file).write_text("last,first\nSmith,John")

        output_file = self.temp_dir / "output.csv"

        # Mock to simulate KeyboardInterrupt
        with patch("ethnicolr2.pred_fl_last_name") as mock_pred:
            mock_pred.side_effect = KeyboardInterrupt("User interrupted")

            result = self.runner.invoke(
                fl_ln_main,
                [input_file, "--output", output_file, "--last-name-col", "last"],
            )

            # Should handle interruption gracefully
            self.assertNotEqual(result.exit_code, 0)


class TestCLIVerboseAndQuietModes(unittest.TestCase):
    """Test CLI verbose and quiet output modes."""

    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        # Create test input file
        self.input_file = self.temp_dir / "test.csv"
        with open(self.input_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["last", "first"])
            writer.writerow(["Smith", "John"])

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_verbose_mode(self):
        """Test verbose output mode."""
        output_file = self.temp_dir / "output.csv"

        result = self.runner.invoke(
            fl_ln_main,
            [
                self.input_file,
                "--output",
                output_file,
                "--last-name-col",
                "last",
                "--verbose",
            ],
        )

        # Should succeed and show verbose output
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Loading data", result.output)
        self.assertIn("Last name column", result.output)

    def test_normal_mode_output(self):
        """Test normal (non-verbose) output mode."""
        output_file = self.temp_dir / "output.csv"

        result = self.runner.invoke(
            fl_ln_main,
            [self.input_file, "--output", output_file, "--last-name-col", "last"],
        )

        # Should succeed with minimal output
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Writing output", result.output)
        # Should not include verbose messages
        self.assertNotIn("Loading data", result.output)

    def test_output_file_creation(self):
        """Test that output files are created correctly."""
        output_file = self.temp_dir / "output.csv"

        result = self.runner.invoke(
            fl_ln_main,
            [self.input_file, "--output", output_file, "--last-name-col", "last"],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertTrue(output_file.exists())

        # Check output file content
        with open(output_file) as f:
            content = f.read()
            self.assertIn("preds", content)  # Should have predictions column
            self.assertIn("Smith", content)  # Should have original data


if __name__ == "__main__":
    unittest.main()
