"""Unit tests for NumPy least squares implementation."""

import os
import sys
import unittest
import warnings

import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from approaches.least_squares_numpy import LeastSquares, LinearRegression, RidgeRegression


class TestLeastSquaresNumpy(unittest.TestCase):
    """Test NumPy-based least squares implementations."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.simple_x = np.array([1, 2, 3, 4, 5])
        self.simple_y = np.array([2, 4, 6, 8, 10])
        self.noise_x = np.linspace(0, 10, 50)
        self.noise_y = 2 * self.noise_x + 1 + np.random.normal(0, 0.5, 50)

    def test_least_squares_init_valid_types(self):
        """Test initialization with valid regression types."""
        valid_types = ["LinearRegression", "RidgeRegression", "LassoRegression", "ElasticNetRegression"]
        for reg_type in valid_types:
            model = LeastSquares(reg_type)
            self.assertEqual(model.type_regression, reg_type)

    def test_least_squares_init_invalid_type(self):
        """Test initialization with invalid regression type."""
        with self.assertRaises(ValueError):
            LeastSquares("InvalidRegression")

    def test_multivariate_ols_basic(self):
        """Test basic multivariate OLS functionality."""
        model = LeastSquares("LinearRegression")
        x_matrix = self.simple_x.reshape(-1, 1)
        coeffs = model.multivariate_ols(x_matrix, self.simple_y)
        self.assertEqual(len(coeffs), 2)
        self.assertAlmostEqual(coeffs[1], 2.0, places=5)

    def test_multivariate_ols_underdetermined(self):
        """Test error handling for underdetermined system."""
        model = LeastSquares("LinearRegression")
        x_small = np.array([[1], [2]])
        y_small = np.array([1, 2])
        x_big_features = np.column_stack([x_small, x_small**2, x_small**3])
        with self.assertRaises(ValueError):
            model.multivariate_ols(x_big_features, y_small)

    def test_qr_decomposition_method(self):
        """Test QR decomposition method."""
        model = LeastSquares("LinearRegression")
        x_matrix = self.noise_x.reshape(-1, 1)
        x_with_intercept = np.column_stack([np.ones(len(self.noise_y)), x_matrix])
        coeffs = model.qr_decomposition(x_with_intercept, self.noise_y)
        self.assertEqual(len(coeffs), 2)
        self.assertAlmostEqual(coeffs[1], 2.0, delta=0.1)

    def test_linear_regression_class(self):
        """Test LinearRegression class functionality."""
        model = LinearRegression(degree=2)
        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([1, 4, 9, 16, 25])
        model.fit(x_data, y_data)
        self.assertIsNotNone(model.coefficients)
        predictions = model.predict(np.array([3]))
        self.assertAlmostEqual(predictions[0], 9, delta=1.0)

    def test_linear_regression_underdetermined(self):
        """Test LinearRegression error handling for insufficient data."""
        model = LinearRegression(degree=5)
        x_small = np.array([1, 2, 3])
        y_small = np.array([1, 4, 9])
        with self.assertRaises(ValueError):
            model.fit(x_small, y_small)

    def test_ridge_regression_class(self):
        """Test RidgeRegression class functionality."""
        model = RidgeRegression(alpha=0.1)
        model.fit(self.noise_x, self.noise_y)
        self.assertIsNotNone(model.coefficients)
        predictions = model.predict(np.array([5.0]))
        self.assertIsInstance(predictions, (list, np.ndarray))
        self.assertGreater(len(predictions), 0)

    def test_polynomial_features_generation(self):
        """Test polynomial feature generation."""
        model = LinearRegression(degree=3)
        test_x = np.array([1, 2, 3])
        features = model._generate_polynomial_features(test_x)
        self.assertEqual(features.shape[1], 3)
        self.assertEqual(features.shape[0], 3)

    def test_condition_number_calculation(self):
        """Test condition number calculation for stability."""
        model = LeastSquares("LinearRegression")
        x_matrix = np.random.randn(20, 3)
        condition_num = model._calculate_standard_condition_number(x_matrix)
        self.assertIsInstance(condition_num, float)
        self.assertGreater(condition_num, 0)

    def test_model_persistence(self):
        """Test that fitted models maintain state."""
        model = LinearRegression(degree=1)
        model.fit(self.simple_x, self.simple_y)
        coeffs_before = model.coefficients.copy()
        model.predict(np.array([6]))
        np.testing.assert_array_equal(coeffs_before, model.coefficients)


class TestNumpyRegressionEdgeCases(unittest.TestCase):
    """Test edge cases for NumPy regression implementations."""

    def test_single_data_point_error(self):
        """Test error handling with single data point."""
        model = LinearRegression(degree=1)
        with self.assertRaises(ValueError):
            model.fit(np.array([1]), np.array([1]))

    def test_identical_x_values(self):
        """Test handling of identical x values."""
        model = LinearRegression(degree=1)
        x_identical = np.array([2, 2, 2, 2, 2])
        y_varied = np.array([1, 2, 3, 4, 5])
        model.fit(x_identical, y_varied)
        predictions = model.predict(np.array([2]))
        self.assertIsInstance(predictions[0], float)

    def test_large_degree_polynomial(self):
        """Test high-degree polynomial handling."""
        model = LinearRegression(degree=6)
        x_large = np.linspace(-1, 1, 20)
        y_large = np.sin(x_large) + np.random.normal(0, 0.01, 20)
        model.fit(x_large, y_large)
        self.assertIsNotNone(model.coefficients)

    def test_zero_target_values(self):
        """Test regression with zero target values."""
        model = LinearRegression(degree=1)
        x_data = np.array([1, 2, 3, 4, 5])
        y_zeros = np.zeros(5)
        model.fit(x_data, y_zeros)
        predictions = model.predict(np.array([3]))
        self.assertAlmostEqual(predictions[0], 0, delta=1e-10)


if __name__ == '__main__':
    unittest.main()