#!/usr/bin/env python3
"""Test that benchmark displays variance correctly."""

import sys
import unittest
from io import StringIO
from utils.timer_regression_engines import display_pipeline_results_table


class TestBenchmarkDisplay(unittest.TestCase):
    """Test benchmark display functionality."""

    def test_variance_display(self):
        """Test that variance is displayed in benchmark results."""
        # Mock pipeline results with variance data
        pipeline_results = {
            "NumPy": {
                'avg': 0.1,  # 100ms
                'min': 0.09,  # 90ms
                'max': 0.11,  # 110ms
                'runs': [0.09, 0.1, 0.11]
            },
            "Pure Python": {
                'avg': 1.0,  # 1s
                'min': 0.8,  # 800ms
                'max': 1.2,  # 1.2s
                'runs': [0.8, 1.0, 1.2]
            }
        }

        # Capture output
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            # Display results (will fail on press enter, but that's ok)
            try:
                display_pipeline_results_table(pipeline_results, num_runs=3, total_combinations=28)
            except (OSError, AttributeError):
                pass  # Expected due to press enter

            # Get output
            output = captured_output.getvalue()

            # Check that variance is displayed
            self.assertIn("Variance", output, "Should show Variance column")
            self.assertIn("±", output, "Should show ± symbol for variance")
            self.assertIn("Average Time", output, "Should show Average Time column")

            # Check that both engines are shown
            self.assertIn("NumPy", output)
            self.assertIn("Pure Python", output)

        finally:
            sys.stdout = sys.__stdout__

    def test_zero_variance_handling(self):
        """Test handling of zero variance."""
        # Mock results with no variance
        pipeline_results = {
            "Test Engine": {
                'avg': 0.1,
                'min': 0.1,
                'max': 0.1,
                'runs': [0.1, 0.1, 0.1]
            }
        }

        # Capture output
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            try:
                display_pipeline_results_table(pipeline_results, num_runs=3, total_combinations=1)
            except (OSError, AttributeError):
                pass

            output = captured_output.getvalue()

            # Should show 0% variance
            self.assertIn("±0.0%", output, "Should show ±0.0% for zero variance")

        finally:
            sys.stdout = sys.__stdout__


if __name__ == "__main__":
    unittest.main()
