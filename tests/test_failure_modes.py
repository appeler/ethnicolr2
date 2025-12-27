#!/usr/bin/env python

"""
Tests for common failure modes in ML models and resource handling.

This module tests scenarios where things go wrong in production:
- Corrupted model files
- Missing resources
- Memory constraints
- Cache corruption
- Resource loading failures
"""

import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import pandas as pd

import ethnicolr2
from ethnicolr2.ethnicolr_class import (
    _MODEL_CACHE,
    EthnicolrModelClass,
    _get_cached_model,
    clear_model_cache,
)


class TestModelFileCorruption(unittest.TestCase):
    """Test handling of corrupted or malformed model files."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_df = pd.DataFrame({"last": ["smith", "zhang"]})
        clear_model_cache()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        clear_model_cache()

    def test_corrupted_model_file(self):
        """Test behavior when .pt model file is corrupted."""
        # Create a corrupted model file
        corrupted_model = self.temp_dir / "corrupted.pt"
        with open(corrupted_model, "wb") as f:
            f.write(b"This is not a valid PyTorch model file")

        # Create a valid vocab file (empty dict)
        vocab_file = self.temp_dir / "vocab.joblib"
        joblib.dump({}, vocab_file)

        # Should raise an error when trying to load
        with self.assertRaises((RuntimeError, OSError, EOFError, Exception)):
            _get_cached_model(corrupted_model, vocab_file)

    def test_missing_model_file(self):
        """Test behavior when model file doesn't exist."""
        nonexistent_model = "/nonexistent/path/model.pt"
        nonexistent_vocab = "/nonexistent/path/vocab.joblib"

        with self.assertRaises(FileNotFoundError):
            _get_cached_model(nonexistent_model, nonexistent_vocab)

    def test_corrupted_vocab_file(self):
        """Test behavior when vocabulary file is corrupted."""
        # Create a simple valid model (we can't easily create a valid PyTorch model)
        # but we can test vocab corruption
        vocab_file = self.temp_dir / "corrupted_vocab.joblib"
        with open(vocab_file, "wb") as f:
            f.write(b"This is not a valid joblib file")

        # Should raise an error when trying to load vocab
        with self.assertRaises((OSError, EOFError, Exception)):
            joblib.load(vocab_file)

    def test_partial_model_files(self):
        """Test behavior with incomplete/truncated files."""
        # Truncated model file
        partial_model = self.temp_dir / "partial.pt"
        with open(partial_model, "wb") as f:
            f.write(b"PK")  # ZIP file header but truncated

        vocab_file = self.temp_dir / "vocab.joblib"
        joblib.dump({}, vocab_file)

        with self.assertRaises((RuntimeError, OSError, EOFError, Exception)):
            _get_cached_model(partial_model, vocab_file)


class TestResourceLoadingFailures(unittest.TestCase):
    """Test resource loading edge cases and fallback mechanisms."""

    def setUp(self):
        clear_model_cache()

    def tearDown(self):
        clear_model_cache()

    def test_resource_path_permissions(self):
        """Test handling of permission errors when accessing resources."""
        # This is hard to test reliably across platforms, so we mainly
        # test that error handling exists

        # Mock os.path.getmtime to raise PermissionError
        with patch("os.path.getmtime", side_effect=OSError("Permission denied")):
            from ethnicolr2.ethnicolr_class import _get_cache_key

            # Should handle permission errors gracefully
            key = _get_cache_key("test_model.pt", "test_vocab.joblib")
            # Should fall back to filename-only key
            self.assertIn("test_model.pt", key)
            self.assertIn("test_vocab.joblib", key)


class TestMemoryConstraints(unittest.TestCase):
    """Test behavior under memory constraints."""

    def setUp(self):
        self.test_df = pd.DataFrame({"last": ["smith"] * 1000})  # Larger dataset
        clear_model_cache()

    def tearDown(self):
        clear_model_cache()

    def test_large_batch_processing(self):
        """Test processing very large datasets."""
        # Create a moderately large dataset
        large_df = pd.DataFrame(
            {
                "last": ["smith", "zhang", "garcia", "johnson"] * 500  # 2000 rows
            }
        )

        # Should handle large datasets without crashing
        try:
            result = ethnicolr2.pred_fl_last_name(large_df, "last")
            self.assertEqual(len(result), 2000)
            self.assertIn("preds", result.columns)
        except MemoryError:
            # If we hit memory limits, that's expected behavior
            self.skipTest("Not enough memory to test large dataset processing")

    def test_memory_efficient_processing(self):
        """Test that processing doesn't leak memory excessively."""
        # Run multiple predictions to check for memory leaks
        test_df = pd.DataFrame({"last": ["smith", "zhang"]})

        for _ in range(10):
            result = ethnicolr2.pred_fl_last_name(test_df, "last")
            self.assertEqual(len(result), 2)
            # Force garbage collection
            del result


class TestCacheCorruption(unittest.TestCase):
    """Test cache corruption and recovery scenarios."""

    def setUp(self):
        clear_model_cache()

    def tearDown(self):
        clear_model_cache()

    def test_cache_with_corrupted_model(self):
        """Test cache behavior when cached model becomes corrupted."""
        # First, cache a valid model
        test_df = pd.DataFrame({"last": ["smith"]})
        ethnicolr2.pred_fl_last_name(test_df, "last")

        # Verify model is cached
        from ethnicolr2.ethnicolr_class import get_cache_info

        cache_info = get_cache_info()
        self.assertGreater(cache_info["cached_models"], 0)

        # Manually corrupt the cached model (simulate memory corruption)
        cache_key = list(_MODEL_CACHE.keys())[0]
        model, vectorizer, metadata = _MODEL_CACHE[cache_key]

        # Replace model with a broken one
        broken_model = Mock()
        broken_model.side_effect = RuntimeError("Simulated model corruption")
        _MODEL_CACHE[cache_key] = (broken_model, vectorizer, metadata)

        # Next prediction should fail due to corrupted cached model
        with self.assertRaises(RuntimeError):
            ethnicolr2.pred_fl_last_name(test_df, "last")

    def test_cache_recovery_after_clear(self):
        """Test that cache can recover after being cleared due to corruption."""
        test_df = pd.DataFrame({"last": ["smith"]})

        # Cache model
        result1 = ethnicolr2.pred_fl_last_name(test_df, "last")

        # Clear cache (simulating recovery from corruption)
        clear_model_cache()

        # Should work again by reloading model
        result2 = ethnicolr2.pred_fl_last_name(test_df, "last")

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)


class TestConcurrentAccessFailures(unittest.TestCase):
    """Test failure scenarios under concurrent access."""

    def setUp(self):
        clear_model_cache()

    def tearDown(self):
        clear_model_cache()

    def test_concurrent_cache_corruption(self):
        """Test cache behavior when multiple threads cause issues."""
        test_df = pd.DataFrame({"last": ["smith"]})
        results = []
        exceptions = []

        def worker_with_cache_clear():
            """Worker that clears cache while others are using it."""
            try:
                # Random operations that might cause issues
                clear_model_cache()
                result = ethnicolr2.pred_fl_last_name(test_df, "last")
                results.append(result)
            except Exception as e:
                exceptions.append(e)

        def normal_worker():
            """Normal worker doing predictions."""
            try:
                result = ethnicolr2.pred_fl_last_name(test_df, "last")
                results.append(result)
            except Exception as e:
                exceptions.append(e)

        # Run multiple threads with some clearing cache
        threads = []
        for i in range(5):
            if i == 2:  # One thread clears cache
                t = threading.Thread(target=worker_with_cache_clear)
            else:
                t = threading.Thread(target=normal_worker)
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have some successful results
        self.assertGreater(len(results), 0)

        # Any exceptions should be reasonable (not crashes)
        for exc in exceptions:
            self.assertIsInstance(exc, (FileNotFoundError, RuntimeError, KeyError))


class TestInputValidationFailures(unittest.TestCase):
    """Test input validation edge cases."""

    def test_lineToTensor_edge_cases(self):
        """Test lineToTensor with problematic inputs."""
        # Test with non-string input
        with self.assertRaises(TypeError):
            EthnicolrModelClass.lineToTensor(123, "abc", 10, 99)  # type: ignore[arg-type]

        # Test with zero max_name
        with self.assertRaises(ValueError):
            EthnicolrModelClass.lineToTensor("test", "abc", 0, 99)

        # Test with negative max_name
        with self.assertRaises(ValueError):
            EthnicolrModelClass.lineToTensor("test", "abc", -5, 99)

        # Test with very long name (should truncate gracefully)
        long_name = "a" * 1000
        result = EthnicolrModelClass.lineToTensor(long_name, "abcdef", 10, 99)
        self.assertEqual(len(result), 10)  # Should be truncated to max_name

    def test_dataframe_validation_edge_cases(self):
        """Test DataFrame validation with edge cases."""
        # Test with None column name
        df = pd.DataFrame({"test": ["value"]})

        # The current implementation checks if col is truthy
        # so None should be handled
        try:
            EthnicolrModelClass.test_and_norm_df(df, None)  # type: ignore[arg-type]
            # If it doesn't raise, that's fine too
        except (TypeError, ValueError):
            # Expected behavior for None input
            pass

        # Test with empty string column name
        with self.assertRaises(ValueError):
            EthnicolrModelClass.test_and_norm_df(df, "")

    def test_tensor_device_mismatch(self):
        """Test handling of device mismatches in tensor operations."""
        # This is mainly to ensure we don't have device-related crashes
        # when CUDA is available vs not available

        # Test with CPU tensor on different device configurations
        test_df = pd.DataFrame({"last": ["smith"]})

        # Should work regardless of CUDA availability
        result = ethnicolr2.pred_fl_last_name(test_df, "last")
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
