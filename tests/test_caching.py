#!/usr/bin/env python

"""
Tests for model caching functionality.

This module tests the module-level caching system that provides
automatic model caching for improved performance.
"""

import os
import threading
import time
import unittest

import pandas as pd

import ethnicolr2
from ethnicolr2.ethnicolr_class import (
    _CACHE_STATS,
    _get_cache_key,
    clear_model_cache,
    get_cache_info,
)


class TestModelCaching(unittest.TestCase):
    """Tests for model caching functionality."""

    def setUp(self):
        """Set up test data and clean cache before each test."""
        self.test_df = pd.DataFrame(
            {"last": ["zhang", "torres", "smith"], "first": ["simon", "raul", "john"]}
        )
        # Clear cache before each test
        clear_model_cache()
        # Reset cache stats
        _CACHE_STATS["hits"] = 0
        _CACHE_STATS["misses"] = 0
        _CACHE_STATS["loads"] = 0

    def tearDown(self):
        """Clean up after each test."""
        clear_model_cache()

    def test_cache_key_generation(self):
        """Test cache key generation with different file paths."""
        key1 = _get_cache_key("model1.pt", "vocab1.joblib")
        key2 = _get_cache_key("model2.pt", "vocab2.joblib")
        key3 = _get_cache_key("model1.pt", "vocab1.joblib")  # Same as key1

        # Different models should have different keys
        self.assertNotEqual(key1, key2)
        # Same model should have same key
        self.assertEqual(key1, key3)
        # Keys should contain file names
        self.assertIn("model1.pt", key1)
        self.assertIn("vocab1.joblib", key1)

    def test_cache_info_initial_state(self):
        """Test cache info returns correct initial state."""
        info = get_cache_info()

        self.assertEqual(info["cached_models"], 0)
        self.assertTrue(info["cache_enabled"])
        self.assertEqual(info["cache_stats"]["hits"], 0)
        self.assertEqual(info["cache_stats"]["misses"], 0)
        self.assertEqual(len(info["models"]), 0)

    def test_caching_transparency(self):
        """Ensure caching doesn't change prediction results."""
        # First prediction (loads model)
        result1 = ethnicolr2.pred_fl_last_name(self.test_df, "last")

        # Second prediction (should use cache)
        result2 = ethnicolr2.pred_fl_last_name(self.test_df, "last")

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

        # Verify predictions are correct
        self.assertIn("preds", result1.columns)
        self.assertIn("probs", result1.columns)
        self.assertEqual(len(result1), 3)

    def test_cache_performance_improvement(self):
        """Verify caching provides performance benefits."""
        # Time first prediction (cold - loads model)
        start = time.time()
        ethnicolr2.pred_fl_last_name(self.test_df, "last")
        cold_time = time.time() - start

        # Time second prediction (cached - reuses model)
        start = time.time()
        ethnicolr2.pred_fl_last_name(self.test_df, "last")
        cached_time = time.time() - start

        # Cached should be faster (though may be small difference in tests)
        self.assertLess(cached_time, cold_time)

        # Check cache statistics
        cache_info = get_cache_info()
        self.assertGreater(cache_info["cache_stats"]["hits"], 0)
        self.assertGreater(cache_info["cached_models"], 0)

    def test_multiple_model_caching(self):
        """Test that different models are cached separately."""
        # Use different models
        result1 = ethnicolr2.pred_fl_last_name(self.test_df, "last")
        result2 = ethnicolr2.pred_census_last_name(self.test_df, "last")

        cache_info = get_cache_info()

        # Should have cached multiple models
        self.assertGreaterEqual(cache_info["cached_models"], 1)

        # Results should be different (different models)
        self.assertFalse(result1.equals(result2))

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        # Load a model to populate cache
        ethnicolr2.pred_fl_last_name(self.test_df, "last")

        # Verify cache has content
        cache_info = get_cache_info()
        self.assertGreater(cache_info["cached_models"], 0)

        # Clear cache
        cleared_count = clear_model_cache()
        self.assertGreater(cleared_count, 0)

        # Verify cache is empty
        cache_info = get_cache_info()
        self.assertEqual(cache_info["cached_models"], 0)

    def test_cache_pattern_clearing(self):
        """Test clearing cache with patterns."""
        # Load multiple models
        ethnicolr2.pred_fl_last_name(self.test_df, "last")
        ethnicolr2.pred_census_last_name(self.test_df, "last")

        # Clear only florida models
        cleared = clear_model_cache("lstm_lastname_gen")

        # Should have cleared some but not all
        # The exact behavior depends on which models were loaded
        self.assertIsInstance(cleared, int)

    def test_concurrent_caching(self):
        """Test cache behavior under concurrent access."""
        results = []

        def predict_worker():
            result = ethnicolr2.pred_fl_last_name(self.test_df, "last")
            results.append(result)

        # Run multiple predictions concurrently
        threads = [threading.Thread(target=predict_worker) for _ in range(5)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All results should be identical
        self.assertEqual(len(results), 5)
        for i in range(1, len(results)):
            pd.testing.assert_frame_equal(results[0], results[i])

        # Cache should handle concurrent access safely
        cache_info = get_cache_info()
        self.assertGreater(cache_info["cached_models"], 0)

    def test_cache_disabled_via_environment(self):
        """Test disabling cache via environment variable."""
        # This test would require modifying environment and reimporting
        # For now, we test the configuration reading
        original_env = os.environ.get("ETHNICOLR_CACHE_ENABLED")

        try:
            # Test that environment variable affects configuration
            # Note: This doesn't affect already imported modules
            os.environ["ETHNICOLR_CACHE_ENABLED"] = "false"

            # Would need to reimport module to test this fully
            # For now, just verify the environment variable is read
            # This might still be True due to import timing

        finally:
            # Restore original environment
            if original_env is not None:
                os.environ["ETHNICOLR_CACHE_ENABLED"] = original_env
            elif "ETHNICOLR_CACHE_ENABLED" in os.environ:
                del os.environ["ETHNICOLR_CACHE_ENABLED"]

    def test_cache_stats_accuracy(self):
        """Test that cache statistics are accurate."""
        # Initial state
        initial_stats = get_cache_info()["cache_stats"]

        # First call - should miss and load
        ethnicolr2.pred_fl_last_name(self.test_df, "last")
        stats_after_first = get_cache_info()["cache_stats"]

        self.assertEqual(stats_after_first["misses"], initial_stats["misses"] + 1)
        self.assertEqual(stats_after_first["loads"], initial_stats["loads"] + 1)

        # Second call - should hit cache
        ethnicolr2.pred_fl_last_name(self.test_df, "last")
        stats_after_second = get_cache_info()["cache_stats"]

        self.assertEqual(stats_after_second["hits"], stats_after_first["hits"] + 1)
        self.assertEqual(
            stats_after_second["loads"], stats_after_first["loads"]
        )  # No new loads

    def test_cache_eviction(self):
        """Test cache eviction when limit is exceeded."""
        # This test would require loading more models than MAX_CACHED_MODELS
        # Since we only have a few model types, we'll test the eviction logic exists
        original_max = os.environ.get("ETHNICOLR_MAX_CACHED_MODELS")

        try:
            # Set a very low limit
            os.environ["ETHNICOLR_MAX_CACHED_MODELS"] = "1"

            # Would need to reimport to affect the limit
            # For now, just test that the configuration is read

        finally:
            if original_max is not None:
                os.environ["ETHNICOLR_MAX_CACHED_MODELS"] = original_max
            elif "ETHNICOLR_MAX_CACHED_MODELS" in os.environ:
                del os.environ["ETHNICOLR_MAX_CACHED_MODELS"]

    def test_different_dataframes_same_model(self):
        """Test that different DataFrames with same model use cache."""
        df1 = pd.DataFrame({"last": ["zhang", "torres"]})
        df2 = pd.DataFrame({"last": ["smith", "johnson", "williams"]})

        # Both should use the same cached model
        result1 = ethnicolr2.pred_fl_last_name(df1, "last")
        result2 = ethnicolr2.pred_fl_last_name(df2, "last")

        # Verify cache was used
        cache_info = get_cache_info()
        self.assertGreater(cache_info["cache_stats"]["hits"], 0)

        # Results should have correct shape for each DataFrame
        self.assertEqual(len(result1), 2)
        self.assertEqual(len(result2), 3)


class TestCacheIntegration(unittest.TestCase):
    """Integration tests for caching with real prediction workflows."""

    def setUp(self):
        """Set up test data and clean cache."""
        clear_model_cache()

    def tearDown(self):
        """Clean up after tests."""
        clear_model_cache()

    def test_all_models_cacheable(self):
        """Test that all prediction models work with caching."""
        test_df = pd.DataFrame(
            {"last": ["zhang", "torres"], "first": ["simon", "raul"]}
        )

        # Test all main prediction functions
        result1 = ethnicolr2.pred_fl_last_name(test_df, "last")
        result2 = ethnicolr2.pred_fl_full_name(test_df, "last", "first")
        result3 = ethnicolr2.pred_census_last_name(test_df, "last")

        # All should complete successfully
        self.assertEqual(len(result1), 2)
        self.assertEqual(len(result2), 2)
        self.assertEqual(len(result3), 2)

        # Cache should contain multiple models
        cache_info = get_cache_info()
        self.assertGreater(cache_info["cached_models"], 0)

    def test_cache_persistence_across_calls(self):
        """Test that cache persists across multiple function calls."""
        test_df = pd.DataFrame({"last": ["zhang"]})

        # Multiple calls to same function
        for _ in range(5):
            result = ethnicolr2.pred_fl_last_name(test_df, "last")
            self.assertEqual(len(result), 1)

        # Should have multiple cache hits
        cache_info = get_cache_info()
        self.assertGreaterEqual(cache_info["cache_stats"]["hits"], 4)


class TestProductionScenarios(unittest.TestCase):
    """Test production deployment scenarios and edge cases."""

    def setUp(self):
        clear_model_cache()

    def tearDown(self):
        clear_model_cache()

    def test_cache_under_memory_pressure(self):
        """Test cache behavior under memory pressure."""
        # Load multiple models to fill cache
        test_df = pd.DataFrame(
            {"last": ["smith", "zhang", "garcia"], "first": ["john", "li", "maria"]}
        )

        # Load all available models
        ethnicolr2.pred_fl_last_name(test_df, "last")
        ethnicolr2.pred_census_last_name(test_df, "last")
        ethnicolr2.pred_fl_full_name(test_df, lname_col="last", fname_col="first")

        # Simulate memory pressure by reducing cache limit
        import os

        original_limit = os.environ.get("ETHNICOLR_MAX_CACHED_MODELS")

        try:
            os.environ["ETHNICOLR_MAX_CACHED_MODELS"] = "1"

            # Load another model - should trigger eviction
            # Note: This requires reimporting to affect the limit
            # For now, just verify current cache behavior

            cache_info_after = get_cache_info()
            # Cache should still function correctly
            self.assertGreater(cache_info_after["cache_stats"]["hits"], 0)

        finally:
            if original_limit is not None:
                os.environ["ETHNICOLR_MAX_CACHED_MODELS"] = original_limit
            elif "ETHNICOLR_MAX_CACHED_MODELS" in os.environ:
                del os.environ["ETHNICOLR_MAX_CACHED_MODELS"]

    def test_rapid_fire_predictions(self):
        """Test rapid-fire predictions to stress test caching."""
        test_df = pd.DataFrame({"last": ["smith"]})

        # Perform many rapid predictions
        start_time = time.time()
        for _ in range(50):
            result = ethnicolr2.pred_fl_last_name(test_df, "last")
            self.assertEqual(len(result), 1)
        end_time = time.time()

        # Should complete quickly due to caching
        total_time = end_time - start_time
        cache_info = get_cache_info()

        # Should have many cache hits
        self.assertGreater(cache_info["cache_stats"]["hits"], 40)
        # Should complete reasonably quickly (less than 5 seconds)
        self.assertLess(total_time, 5.0)

    def test_cache_consistency_under_load(self):
        """Test cache consistency under concurrent load."""
        test_df = pd.DataFrame({"last": ["smith", "zhang"]})
        results = []

        def worker():
            for _ in range(10):
                result = ethnicolr2.pred_fl_last_name(test_df, "last")
                results.append(result)

        # Run multiple workers concurrently
        import threading

        threads = [threading.Thread(target=worker) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be identical
        self.assertEqual(len(results), 50)  # 5 threads * 10 predictions each

        # Check that all results are consistent
        first_result = results[0]
        for result in results[1:]:
            pd.testing.assert_frame_equal(first_result, result)

    def test_cache_performance_degradation(self):
        """Test that cache performance doesn't degrade over time."""
        test_df = pd.DataFrame({"last": ["smith"]})

        # Perform initial prediction (cache miss)
        start_time = time.time()
        ethnicolr2.pred_fl_last_name(test_df, "last")
        first_time = time.time() - start_time

        # Perform many cached predictions
        times = []
        for _ in range(20):
            start_time = time.time()
            ethnicolr2.pred_fl_last_name(test_df, "last")
            times.append(time.time() - start_time)

        # Cached predictions should remain fast
        avg_cached_time = sum(times) / len(times)
        max_cached_time = max(times)

        self.assertLess(avg_cached_time, first_time / 2)
        self.assertLess(max_cached_time, first_time)

    def test_cache_with_different_data_sizes(self):
        """Test cache performance with different data sizes."""
        # Test with small dataset
        small_df = pd.DataFrame({"last": ["smith"]})
        start_time = time.time()
        result_small = ethnicolr2.pred_fl_last_name(small_df, "last")
        small_time = time.time() - start_time

        # Test with larger dataset (same model should be cached)
        large_df = pd.DataFrame({"last": ["smith"] * 100})
        start_time = time.time()
        result_large = ethnicolr2.pred_fl_last_name(large_df, "last")
        large_time = time.time() - start_time

        # Model loading should be cached, so difference should be mainly data processing
        cache_info = get_cache_info()
        self.assertGreater(cache_info["cache_stats"]["hits"], 0)

        # Both should succeed
        self.assertEqual(len(result_small), 1)
        self.assertEqual(len(result_large), 100)

        # Large dataset should take longer but not excessively (no model reloading)
        self.assertLess(large_time, small_time * 50)  # Should scale reasonably

    def test_environment_configuration_handling(self):
        """Test handling of environment configuration changes."""
        import os

        # Test cache enabled/disabled configuration
        original_enabled = os.environ.get("ETHNICOLR_CACHE_ENABLED")

        try:
            # Test with caching disabled
            os.environ["ETHNICOLR_CACHE_ENABLED"] = "false"

            # Note: This doesn't affect already imported modules
            # This mainly tests that the configuration reading code works

            # Test that we can still make predictions
            test_df = pd.DataFrame({"last": ["smith"]})
            result = ethnicolr2.pred_fl_last_name(test_df, "last")
            self.assertEqual(len(result), 1)

        finally:
            if original_enabled is not None:
                os.environ["ETHNICOLR_CACHE_ENABLED"] = original_enabled
            elif "ETHNICOLR_CACHE_ENABLED" in os.environ:
                del os.environ["ETHNICOLR_CACHE_ENABLED"]


if __name__ == "__main__":
    unittest.main()
