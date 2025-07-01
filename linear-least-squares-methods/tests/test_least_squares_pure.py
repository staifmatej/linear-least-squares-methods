"""Test cases for least_squares_pure.py module."""

import unittest
import numpy as np
from approaches.least_squares_pure import LinearRegression, RidgeRegression


class TestLeastSquaresPure(unittest.TestCase):
    """Test cases for Pure Python least squares implementations."""

    def setUp(self):
        """Set up test data."""
        # Simple linear data
        self.X_simple = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_simple = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # y = 2x

        # Quadratic data
        self.X_quad = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_quad = np.array([1.0, 4.0, 9.0, 16.0, 25.0])  # y = x^2

    def test_linear_regression_init(self):
        """Test LinearRegression initialization."""
        model = LinearRegression()
        self.assertEqual(model.degree, 1)
        self.assertIsNone(model.coefficients)

        model_deg3 = LinearRegression(degree=3)
        self.assertEqual(model_deg3.degree, 3)

    def test_linear_regression_fit_simple(self):
        """Test LinearRegression fit with simple linear data."""
        model = LinearRegression(degree=1)
        model.fit(self.X_simple, self.y_simple)

        self.assertIsNotNone(model.coefficients)
        self.assertEqual(len(model.coefficients), 2)  # intercept + slope

        # Check if coefficients are approximately correct (intercept≈0, slope≈2)
        self.assertAlmostEqual(model.coefficients[0], 0.0, places=1)  # intercept
        self.assertAlmostEqual(model.coefficients[1], 2.0, places=1)  # slope

    def test_linear_regression_predict(self):
        """Test LinearRegression prediction."""
        model = LinearRegression(degree=1)
        model.fit(self.X_simple, self.y_simple)

        # Test prediction
        X_test = [6.0]  # Pure python expects flat list
        predictions = model.predict(X_test)

        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)
        # Should predict approximately 12.0 for x=6
        self.assertAlmostEqual(predictions[0], 12.0, places=0)

    def test_quadratic_regression_fit(self):
        """Test quadratic regression fitting."""
        model = LinearRegression(degree=2)
        model.fit(self.X_quad, self.y_quad)

        self.assertIsNotNone(model.coefficients)
        self.assertEqual(len(model.coefficients), 3)  # intercept + x + x^2

    def test_ridge_regression_init(self):
        """Test RidgeRegression initialization."""
        model = RidgeRegression()
        self.assertEqual(model.alpha, 1.0)
        self.assertIsNone(model.coefficients)

        model_alpha = RidgeRegression(alpha=0.5)
        self.assertEqual(model_alpha.alpha, 0.5)

    def test_ridge_regression_fit(self):
        """Test RidgeRegression fitting."""
        model = RidgeRegression(alpha=0.1)
        model.fit(self.X_simple, self.y_simple)

        self.assertIsNotNone(model.coefficients)
        # Ridge should have coefficients
        self.assertGreater(len(model.coefficients), 0)

    def test_ridge_regression_predict(self):
        """Test RidgeRegression prediction."""
        model = RidgeRegression(alpha=0.1)
        model.fit(self.X_simple, self.y_simple)

        X_test = [3.0]  # Pure python expects flat list
        predictions = model.predict(X_test)

        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions[0], (int, float))

    def test_error_handling_insufficient_data(self):
        """Test error handling with insufficient data."""
        model = LinearRegression(degree=5)  # High degree
        X_small = [1.0, 2.0]  # Only 2 points
        y_small = [1.0, 2.0]

        # Should handle underdetermined system gracefully
        try:
            model.fit(X_small, y_small)
            # If it doesn't raise an error, check that coefficients exist
            self.assertIsNotNone(model.coefficients)
        except (ValueError, RuntimeError):
            # It's acceptable to raise an error for underdetermined systems
            pass

    def test_model_consistency(self):
        """Test that model predictions are consistent."""
        model = LinearRegression(degree=1)
        model.fit(self.X_simple, self.y_simple)

        # Same input should give same output
        X_test = [3.0]  # Pure python expects flat list
        pred1 = model.predict(X_test)
        pred2 = model.predict(X_test)

        self.assertEqual(pred1, pred2)

    def test_different_degrees(self):
        """Test fitting with different polynomial degrees."""
        for degree in [1, 2, 3]:
            with self.subTest(degree=degree):
                model = LinearRegression(degree=degree)
                model.fit(self.X_simple, self.y_simple)

                self.assertIsNotNone(model.coefficients)
                self.assertEqual(len(model.coefficients), degree + 1)


if __name__ == '__main__':
    unittest.main()
