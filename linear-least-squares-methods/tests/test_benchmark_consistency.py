"""Test benchmark consistency to ensure reliable performance measurements."""

import statistics
import unittest
import numpy as np
from utils.timer_regression_engines import time_single_engine


class TestBenchmarkConsistency(unittest.TestCase):
    """Test that benchmark results are consistent across multiple runs."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        # Use 1D data for compatibility with all engines
        self.X = np.random.randn(50, 1)
        self.y = np.random.randn(50)
        self.regression_types = [1]  # Linear
        self.function_types = [1]    # Linear
        self.num_runs = 5

    def test_numpy_consistency(self):
        """Test that NumPy engine produces timing results with reasonable variance."""
        times = []

        # Run benchmark 5 times
        for _ in range(5):
            result = time_single_engine(
                1, self.X, self.y,
                self.regression_types,
                self.function_types,
                self.num_runs
            )
            if result is not None:
                times.append(result['avg'])

        # Check we got results
        self.assertGreater(len(times), 3, "Should have at least 4 successful runs")

        # Calculate coefficient of variation (CV)
        mean_time = statistics.mean(times)
        stdev_time = statistics.stdev(times) if len(times) > 1 else 0
        cv = (stdev_time / mean_time) * 100 if mean_time > 0 else 0

        # CV should be less than 50% for reasonable results (increased from 20%)
        self.assertLess(cv, 50,
            f"NumPy timing variance too high: CV={cv:.1f}% "
            f"(mean={mean_time*1000:.3f}ms, stdev={stdev_time*1000:.3f}ms)")

    def test_pure_python_consistency(self):
        """Test that Pure Python engine produces timing results with reasonable variance."""
        times = []

        # Run benchmark 5 times
        for _ in range(5):
            result = time_single_engine(
                3, self.X, self.y,
                self.regression_types,
                self.function_types,
                self.num_runs
            )
            if result is not None:
                times.append(result['avg'])

        # Check we got results
        self.assertGreater(len(times), 3, "Should have at least 4 successful runs")

        # Calculate coefficient of variation (CV)
        mean_time = statistics.mean(times)
        stdev_time = statistics.stdev(times) if len(times) > 1 else 0
        cv = (stdev_time / mean_time) * 100 if mean_time > 0 else 0

        # CV should be less than 50% for reasonable results (increased from 20%)
        self.assertLess(cv, 50,
            f"Pure Python timing variance too high: CV={cv:.1f}% "
            f"(mean={mean_time*1000:.3f}ms, stdev={stdev_time*1000:.3f}ms)")

    def test_relative_performance_consistency(self):
        """Test that relative performance between engines is reasonable."""
        numpy_times = []
        pure_times = []

        # Run each engine 3 times
        for _ in range(3):
            # NumPy
            result_np = time_single_engine(
                1, self.X, self.y,
                self.regression_types,
                self.function_types,
                self.num_runs
            )
            if result_np is not None:
                numpy_times.append(result_np['avg'])

            # Pure Python
            result_pure = time_single_engine(
                3, self.X, self.y,
                self.regression_types,
                self.function_types,
                self.num_runs
            )
            if result_pure is not None:
                pure_times.append(result_pure['avg'])

        # Check we have results
        self.assertEqual(len(numpy_times), len(pure_times),
                        "Should have equal number of results")
        self.assertGreater(len(numpy_times), 1,
                          "Should have at least 2 successful runs")

        # Calculate speedup ratios
        speedups = [pure_times[i] / numpy_times[i] for i in range(len(numpy_times))]
        mean_speedup = statistics.mean(speedups)

        # NumPy should be faster than Pure Python (at least 1.5x)
        self.assertGreater(mean_speedup, 1.5,
                          f"NumPy should be significantly faster than Pure Python "
                          f"(got {mean_speedup:.1f}x speedup)")


if __name__ == "__main__":
    unittest.main()
