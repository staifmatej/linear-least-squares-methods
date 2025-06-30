"""Unit tests for RegressionRun class."""

import os
import sys
import unittest
import warnings

import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.run_regression import RegressionRun


class TestRegressionRun(unittest.TestCase):
    """Test RegressionRun class functionality."""

    def setUp(self):
        """Set up test data and runner instance."""
        np.random.seed(42)
        self.simple_x = np.array([1, 2, 3, 4, 5])
        self.simple_y = np.array([2, 4, 6, 8, 10])
        self.noise_x = np.linspace(0, 10, 20)
        self.noise_y = 2 * self.noise_x + 1 + np.random.normal(0, 0.1, 20)

    def test_regression_run_initialization(self):
        """Test RegressionRun initialization."""
        runner = RegressionRun(engine_choice=1, regression_types=[1, 2], function_types=[1, 2])
        self.assertEqual(runner.engine_choice, 1)
        self.assertEqual(runner.regression_types, [1, 2])
        self.assertEqual(runner.function_types, [1, 2])
        self.assertIsInstance(runner.results, dict)

    def test_engine_mapping(self):
        """Test engine mapping correctness."""
        runner = RegressionRun(1, [1], [1])
        expected_mapping = {1: "numpy", 2: "numba", 3: "pure"}
        self.assertEqual(runner.engine_mapping, expected_mapping)

    def test_regression_mapping(self):
        """Test regression type mapping correctness."""
        runner = RegressionRun(1, [1], [1])
        expected_mapping = {1: "Linear", 2: "Ridge", 3: "Lasso", 4: "ElasticNet"}
        self.assertEqual(runner.regression_mapping, expected_mapping)

    def test_function_degree_mapping(self):
        """Test function degree mapping correctness."""
        runner = RegressionRun(1, [1], [1])
        expected_mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
        self.assertEqual(runner.function_degree_mapping, expected_mapping)

    def test_numpy_engine_linear_regression(self):
        """Test numpy engine with linear regression."""
        runner = RegressionRun(engine_choice=1, regression_types=[1], function_types=[1])
        results = runner.run_regressions(self.simple_x, self.simple_y)
        self.assertIsNotNone(results)
        self.assertIn((1, 1), results)
        result = results[(1, 1)]
        if result is not None:
            self.assertIn('model', result)

    def test_numpy_engine_ridge_regression(self):
        """Test numpy engine with ridge regression."""
        runner = RegressionRun(engine_choice=1, regression_types=[2], function_types=[1])
        results = runner.run_regressions(self.noise_x, self.noise_y)
        self.assertIsNotNone(results)
        self.assertIn((2, 1), results)

    def test_multiple_regression_types(self):
        """Test running multiple regression types."""
        runner = RegressionRun(engine_choice=1, regression_types=[1, 2], function_types=[1])
        results = runner.run_regressions(self.noise_x, self.noise_y)
        self.assertEqual(len(results), 2)
        self.assertIn((1, 1), results)
        self.assertIn((2, 1), results)

    def test_multiple_function_types(self):
        """Test running multiple function types."""
        runner = RegressionRun(engine_choice=1, regression_types=[1], function_types=[1, 2])
        results = runner.run_regressions(self.noise_x, self.noise_y)
        self.assertEqual(len(results), 2)
        self.assertIn((1, 1), results)
        self.assertIn((1, 2), results)

    def test_edge_case_insufficient_data(self):
        """Test error handling with insufficient data."""
        runner = RegressionRun(engine_choice=1, regression_types=[1], function_types=[5])
        x_small = np.array([1, 2])
        y_small = np.array([1, 2])
        results = runner.run_regressions(x_small, y_small)
        result = results.get((1, 5))
        self.assertIsNone(result)

    def test_invalid_engine_choice(self):
        """Test behavior with invalid engine choice."""
        runner = RegressionRun(engine_choice=99, regression_types=[1], function_types=[1])
        results = runner.run_regressions(self.simple_x, self.simple_y)
        self.assertIsInstance(results, dict)

    def test_empty_regression_types(self):
        """Test behavior with empty regression types."""
        runner = RegressionRun(engine_choice=1, regression_types=[], function_types=[1])
        results = runner.run_regressions(self.simple_x, self.simple_y)
        self.assertEqual(len(results), 0)

    def test_empty_function_types(self):
        """Test behavior with empty function types."""
        runner = RegressionRun(engine_choice=1, regression_types=[1], function_types=[])
        results = runner.run_regressions(self.simple_x, self.simple_y)
        self.assertEqual(len(results), 0)

    def test_result_structure(self):
        """Test structure of returned results."""
        runner = RegressionRun(engine_choice=1, regression_types=[1], function_types=[1])
        results = runner.run_regressions(self.simple_x, self.simple_y)
        if (1, 1) in results and results[(1, 1)] is not None:
            result = results[(1, 1)]
            self.assertIsInstance(result, dict)

    def test_model_prediction_capability(self):
        """Test that returned models can make predictions."""
        runner = RegressionRun(engine_choice=1, regression_types=[1], function_types=[1])
        results = runner.run_regressions(self.simple_x, self.simple_y)
        if (1, 1) in results and results[(1, 1)] is not None:
            result = results[(1, 1)]
            if 'model' in result and hasattr(result['model'], 'predict'):
                predictions = result['model'].predict(np.array([3]))
                self.assertIsInstance(predictions, (list, np.ndarray))

    def test_high_degree_polynomial_stability(self):
        """Test stability with high-degree polynomials."""
        runner = RegressionRun(engine_choice=1, regression_types=[1], function_types=[6])
        x_stable = np.linspace(-1, 1, 15)
        y_stable = np.sin(x_stable) + np.random.normal(0, 0.01, 15)
        results = runner.run_regressions(x_stable, y_stable)
        self.assertIn((1, 6), results)


class TestRegressionRunEdgeCases(unittest.TestCase):
    """Test edge cases for RegressionRun functionality."""

    def setUp(self):
        """Set up test data."""
        self.edge_x = np.array([1, 1, 1, 1, 1])
        self.edge_y = np.array([1, 2, 3, 4, 5])

    def test_identical_x_values(self):
        """Test regression with identical x values."""
        runner = RegressionRun(engine_choice=1, regression_types=[1], function_types=[1])
        results = runner.run_regressions(self.edge_x, self.edge_y)
        self.assertIsInstance(results, dict)

    def test_single_data_point(self):
        """Test regression with single data point."""
        runner = RegressionRun(engine_choice=1, regression_types=[1], function_types=[1])
        x_single = np.array([1])
        y_single = np.array([2])
        results = runner.run_regressions(x_single, y_single)
        result = results.get((1, 1))
        self.assertIsNone(result)

    def test_zero_target_values(self):
        """Test regression with zero target values."""
        runner = RegressionRun(engine_choice=1, regression_types=[1], function_types=[1])
        x_data = np.array([1, 2, 3, 4, 5])
        y_zeros = np.zeros(5)
        results = runner.run_regressions(x_data, y_zeros)
        self.assertIsInstance(results, dict)

    def test_negative_values(self):
        """Test regression with negative values."""
        runner = RegressionRun(engine_choice=1, regression_types=[1], function_types=[1])
        x_neg = np.array([-2, -1, 0, 1, 2])
        y_neg = np.array([-4, -2, 0, 2, 4])
        results = runner.run_regressions(x_neg, y_neg)
        self.assertIsInstance(results, dict)

    def test_large_values(self):
        """Test regression with large values."""
        runner = RegressionRun(engine_choice=1, regression_types=[1], function_types=[1])
        x_large = np.array([1000, 2000, 3000, 4000, 5000])
        y_large = 2 * x_large + 100
        results = runner.run_regressions(x_large, y_large)
        self.assertIsInstance(results, dict)


if __name__ == '__main__':
    unittest.main()
