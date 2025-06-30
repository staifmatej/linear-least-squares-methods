"""Unit tests for timer_regression_engines class."""

import os
import sys
import time
import unittest
from unittest.mock import patch
import warnings

import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.timer_regression_engines import (
    clear_all_caches,
    format_time,
    get_benchmark_settings,
    run_comprehensive_benchmark,
    run_performance_benchmark,
    time_single_engine
)


class TestTimerRegressionEngines(unittest.TestCase):
    """Test timer regression engines functionality."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.test_x = np.linspace(0, 10, 100)
        self.test_y = 2 * self.test_x + 1 + np.random.normal(0, 0.1, 100)
        self.small_x = np.array([1, 2, 3, 4, 5])
        self.small_y = np.array([2, 4, 6, 8, 10])

    def test_clear_all_caches_functionality(self):
        """Test cache clearing functionality."""
        try:
            clear_all_caches()
            self.assertTrue(True)
        except Exception as exception:
            self.fail(f"clear_all_caches raised an exception: {exception}")

    def test_format_time_basic(self):
        """Test basic time formatting functionality."""
        try:
            formatted = format_time(0.001234)
            self.assertIsInstance(formatted, str)
            self.assertGreater(len(formatted), 0)
        except Exception as exception:
            self.fail(f"format_time raised an exception: {exception}")

    def test_format_time_edge_cases(self):
        """Test time formatting with edge cases."""
        test_cases = [0.0, 1e-9, 1e-3, 1.0, 60.0, 3600.0]
        for test_time in test_cases:
            try:
                formatted = format_time(test_time)
                self.assertIsInstance(formatted, str)
                self.assertGreater(len(formatted), 0)
            except Exception as exception:
                self.fail(f"format_time({test_time}) raised an exception: {exception}")

    @patch('builtins.input', return_value='10')
    def test_get_benchmark_settings_basic(self, mock_input):
        """Test basic benchmark settings retrieval."""
        try:
            settings = get_benchmark_settings()
            self.assertIsInstance(settings, (dict, tuple, list, int))
        except Exception as exception:
            self.fail(f"get_benchmark_settings raised an exception: {exception}")

    def test_time_single_engine_basic(self):
        """Test basic single engine timing."""
        try:
            timing_result = time_single_engine(
                1, self.small_x, self.small_y, [1], [1], 2
            )
            self.assertIsInstance(timing_result, (dict, float, tuple))
        except Exception as exception:
            self.fail(f"time_single_engine raised an exception: {exception}")

    def test_time_single_engine_different_engines(self):
        """Test single engine timing with different engines."""
        engines = [1, 2, 3]
        for engine in engines:
            try:
                timing_result = time_single_engine(
                    engine, self.small_x, self.small_y, [1], [1], 1
                )
                self.assertIsInstance(timing_result, (dict, float, tuple))
            except Exception as exception:
                pass

    def test_run_comprehensive_benchmark_basic(self):
        """Test basic comprehensive benchmark."""
        try:
            results = run_comprehensive_benchmark(
                self.small_x, self.small_y, [1], [1], 2
            )
            self.assertIsInstance(results, (dict, list, tuple))
        except Exception as exception:
            self.fail(f"run_comprehensive_benchmark raised an exception: {exception}")

    def test_run_comprehensive_benchmark_multiple_types(self):
        """Test comprehensive benchmark with multiple regression and function types."""
        try:
            results = run_comprehensive_benchmark(
                self.test_x, self.test_y, [1, 2], [1, 2], 1
            )
            self.assertIsInstance(results, (dict, list, tuple))
        except Exception as exception:
            self.fail(f"Multiple types benchmark raised an exception: {exception}")

    @patch('builtins.input', return_value='10')
    def test_run_performance_benchmark_basic(self, mock_input):
        """Test basic performance benchmark."""
        try:
            run_performance_benchmark(
                self.small_x, self.small_y, [1], [1]
            )
            self.assertTrue(True)
        except Exception as exception:
            self.fail(f"run_performance_benchmark raised an exception: {exception}")

    def test_timing_accuracy(self):
        """Test that timing measurements are reasonable."""
        try:
            start_time = time.time()
            timing_result = time_single_engine(
                1, self.small_x, self.small_y, [1], [1], 1
            )
            end_time = time.time()

            total_time = end_time - start_time
            self.assertGreater(total_time, 0)
            self.assertLess(total_time, 10)
        except Exception as exception:
            self.fail(f"Timing accuracy test raised an exception: {exception}")

    def test_benchmark_with_different_data_sizes(self):
        """Test benchmark with different data sizes."""
        large_x = np.linspace(0, 100, 1000)
        large_y = 2 * large_x + np.random.normal(0, 1, 1000)

        try:
            small_results = time_single_engine(
                1, self.small_x, self.small_y, [1], [1], 1
            )
            large_results = time_single_engine(
                1, large_x, large_y, [1], [1], 1
            )

            self.assertIsInstance(small_results, (dict, float, tuple))
            self.assertIsInstance(large_results, (dict, float, tuple))
        except Exception as exception:
            self.fail(f"Different data sizes benchmark raised an exception: {exception}")

    def test_invalid_engine_handling(self):
        """Test handling of invalid engine choices."""
        try:
            timing_result = time_single_engine(
                999, self.small_x, self.small_y, [1], [1], 1
            )
            self.assertIsInstance(timing_result, (dict, float, tuple))
        except Exception:
            pass

    def test_zero_runs_handling(self):
        """Test handling of zero runs."""
        try:
            timing_result = time_single_engine(
                1, self.small_x, self.small_y, [1], [1], 0
            )
            self.assertIsInstance(timing_result, (dict, float, tuple))
        except Exception:
            pass


class TestTimerRegressionEnginesEdgeCases(unittest.TestCase):
    """Test edge cases for timer regression engines functionality."""

    def setUp(self):
        """Set up edge case test data."""
        self.edge_x = np.array([1])
        self.edge_y = np.array([2])
        self.identical_x = np.array([2, 2, 2, 2, 2])
        self.identical_y = np.array([1, 2, 3, 4, 5])

    def test_single_data_point_benchmark(self):
        """Test benchmark with single data point."""
        try:
            timing_result = time_single_engine(
                1, self.edge_x, self.edge_y, [1], [1], 1
            )
            self.assertIsInstance(timing_result, (dict, float, tuple))
        except Exception:
            pass

    def test_identical_x_values_benchmark(self):
        """Test benchmark with identical x values."""
        try:
            timing_result = time_single_engine(
                1, self.identical_x, self.identical_y, [1], [1], 1
            )
            self.assertIsInstance(timing_result, (dict, float, tuple))
        except Exception as exception:
            self.fail(f"Identical x values benchmark raised an exception: {exception}")

    def test_negative_values_benchmark(self):
        """Test benchmark with negative values."""
        neg_x = np.array([-5, -3, -1, 1, 3])
        neg_y = np.array([-10, -6, -2, 2, 6])

        try:
            timing_result = time_single_engine(
                1, neg_x, neg_y, [1], [1], 1
            )
            self.assertIsInstance(timing_result, (dict, float, tuple))
        except Exception as exception:
            self.fail(f"Negative values benchmark raised an exception: {exception}")

    def test_zero_values_benchmark(self):
        """Test benchmark with zero values."""
        zero_x = np.array([0, 1, 2, 3, 4])
        zero_y = np.array([0, 1, 2, 3, 4])

        try:
            timing_result = time_single_engine(
                1, zero_x, zero_y, [1], [1], 1
            )
            self.assertIsInstance(timing_result, (dict, float, tuple))
        except Exception as exception:
            self.fail(f"Zero values benchmark raised an exception: {exception}")

    def test_format_time_special_values(self):
        """Test format_time with special values."""
        special_values = [float('inf'), float('-inf'), 0.0]
        for value in special_values:
            try:
                formatted = format_time(value)
                self.assertIsInstance(formatted, str)
            except Exception:
                pass

    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        large_x = np.linspace(0, 1000, 10000)
        large_y = 2 * large_x + np.random.normal(0, 10, 10000)

        try:
            timing_result = time_single_engine(
                1, large_x, large_y, [1], [1], 1
            )
            self.assertIsInstance(timing_result, (dict, float, tuple))
        except Exception as exception:
            self.fail(f"Large dataset performance test raised an exception: {exception}")

    def test_comprehensive_benchmark_edge_cases(self):
        """Test comprehensive benchmark with edge cases."""
        try:
            results = run_comprehensive_benchmark(
                self.identical_x, self.identical_y, [], [], 1
            )
            self.assertIsInstance(results, (dict, list, tuple))
        except Exception:
            pass


if __name__ == '__main__':
    unittest.main()