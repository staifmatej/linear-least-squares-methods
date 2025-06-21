"""Unit tests for LeastSquares class from least_squares_numpy.py"""

import numpy as np
from approaches.least_squares_numpy import LeastSquares, PolynomialRegression


# pylint: disable=attribute-defined-outside-init
class TestOLS:
    """Test cases for LeastSquares class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.ols = LeastSquares()

        # Create simple test data
        self.X_simple = np.array([[1], [2], [3], [4], [5]])
        self.y_simple = np.array([2, 4, 6, 8, 10])  # y = 2x

        # Create more complex test data: y = 1 + 2x1 + 3x2 (avoiding multicollinearity)
        self.X_multi = np.array([[1, 1], [2, 1], [1, 2], [3, 2], [2, 3]])
        self.y_multi = np.array([6, 7, 8, 11, 13])  # y = 1 + 2x1 + 3x2

    def test_ols_initialization(self):
        """Test LeastSquares can be initialized."""
        ols = LeastSquares()
        assert ols is not None
        assert isinstance(ols, LeastSquares)

    def test_multivariate_ols_simple_linear(self):
        """Test LeastSquares with simple linear relationship."""
        coefficients = self.ols.multivariate_ols(self.X_simple, self.y_simple)

        # Should return coefficients [intercept, slope]
        assert len(coefficients) == 2
        assert isinstance(coefficients, np.ndarray)

        # For y = 2x, expect intercept ≈ 0, slope ≈ 2
        np.testing.assert_allclose(coefficients[0], 0, atol=1e-10)  # intercept
        np.testing.assert_allclose(coefficients[1], 2, atol=1e-10)  # slope

    def test_multivariate_ols_multivariable(self):
        """Test LeastSquares with multiple variables."""
        coefficients = self.ols.multivariate_ols(self.X_multi, self.y_multi)

        # Should return coefficients [intercept, coef1, coef2]
        assert len(coefficients) == 3
        assert isinstance(coefficients, np.ndarray)

        # Test that predictions are accurate
        X_with_intercept = np.column_stack([np.ones(len(self.y_multi)), self.X_multi])
        predictions = X_with_intercept @ coefficients

        # Should predict reasonably well
        mse = np.mean((self.y_multi - predictions) ** 2)
        assert mse < 1.0, f"MSE too high: {mse}"  # Reasonable tolerance

        # Test that coefficients are reasonable
        assert all(np.isfinite(coefficients)), "Coefficients should be finite"

    def test_normal_equations_method(self):
        """Test normal equations method directly."""
        # Add intercept column manually for direct method testing
        X_with_intercept = np.column_stack([np.ones(len(self.y_simple)), self.X_simple])

        coefficients = self.ols.normal_equations(X_with_intercept, self.y_simple)

        assert len(coefficients) == 2
        np.testing.assert_allclose(coefficients[0], 0, atol=1e-10)
        np.testing.assert_allclose(coefficients[1], 2, atol=1e-10)

    def test_qr_decomposition_method(self):
        """Test QR decomposition method directly."""
        X_with_intercept = np.column_stack([np.ones(len(self.y_simple)), self.X_simple])

        coefficients = self.ols.qr_decomposition(X_with_intercept, self.y_simple)

        assert len(coefficients) == 2
        np.testing.assert_allclose(coefficients[0], 0, atol=1e-10)
        np.testing.assert_allclose(coefficients[1], 2, atol=1e-10)

    def test_normal_vs_qr_consistency(self):
        """Test that normal equations and QR give same results."""
        X_with_intercept = np.column_stack([np.ones(len(self.y_multi)), self.X_multi])

        coef_normal = self.ols.normal_equations(X_with_intercept, self.y_multi)
        coef_qr = self.ols.qr_decomposition(X_with_intercept, self.y_multi)

        np.testing.assert_allclose(coef_normal, coef_qr, atol=1e-10)

    def test_singular_matrix_handling(self):
        """Test handling of singular matrices."""
        # Create singular matrix (linearly dependent columns)
        X_singular = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])  # Second column = 2 * first column
        y_singular = np.array([1, 2, 3, 4])

        # Should not raise exception and return some result
        coefficients = self.ols.multivariate_ols(X_singular, y_singular)
        assert coefficients is not None
        assert len(coefficients) == 3  # intercept + 2 features
        assert all(np.isfinite(coefficients)), "Coefficients should be finite"

    def test_zero_matrix_input(self):
        """Test with zero matrix."""
        X_zero = np.zeros((4, 2))  # All zeros
        y_zero = np.array([1, 2, 3, 4])

        coefficients = self.ols.multivariate_ols(X_zero, y_zero)
        assert coefficients is not None
        assert len(coefficients) == 3

        # For zero X, best fit should be just the mean of y as intercept
        expected_intercept = np.mean(y_zero)
        assert abs(coefficients[0] - expected_intercept) < 1e-10

    def test_identical_rows(self):
        """Test with identical rows in X."""
        X_identical = np.array([[1, 2], [1, 2], [1, 2], [1, 2]])
        y_identical = np.array([5, 6, 7, 8])

        coefficients = self.ols.multivariate_ols(X_identical, y_identical)
        assert coefficients is not None
        assert all(np.isfinite(coefficients))

    def test_rank_deficient_matrix(self):
        """Test with rank deficient matrix."""
        # Matrix with rank 1 (all rows are multiples of first row)
        X_rank_def = np.array([[1, 2], [2, 4], [3, 6]])
        y_rank_def = np.array([1, 2, 3])

        coefficients = self.ols.multivariate_ols(X_rank_def, y_rank_def)
        assert coefficients is not None
        assert all(np.isfinite(coefficients))

    def test_very_small_values(self):
        """Test with very small values."""
        X_tiny = np.array([[1e-15, 2e-15], [3e-15, 4e-15], [5e-15, 6e-15]])
        y_tiny = np.array([1e-15, 2e-15, 3e-15])

        coefficients = self.ols.multivariate_ols(X_tiny, y_tiny)
        assert coefficients is not None
        assert all(np.isfinite(coefficients))

    def test_very_large_values(self):
        """Test with very large values."""
        X_huge = np.array([[1e15, 2e15], [3e15, 4e15], [5e15, 6e15]])
        y_huge = np.array([1e15, 2e15, 3e15])

        coefficients = self.ols.multivariate_ols(X_huge, y_huge)
        assert coefficients is not None
        assert all(np.isfinite(coefficients))

    def test_mixed_positive_negative(self):
        """Test with mixed positive and negative values."""
        X_mixed = np.array([[-1, 2], [3, -4], [-5, 6], [7, -8]])
        y_mixed = np.array([-1, 2, -3, 4])

        coefficients = self.ols.multivariate_ols(X_mixed, y_mixed)
        assert coefficients is not None
        assert all(np.isfinite(coefficients))

    def test_minimum_data_points(self):
        """Test with minimum required data points."""
        # For 2 parameters (1 feature + intercept), need at least 2 points
        X_min = np.array([[1], [2]])
        y_min = np.array([1, 2])

        coefficients = self.ols.multivariate_ols(X_min, y_min)
        assert coefficients is not None
        assert len(coefficients) == 2
        assert all(np.isfinite(coefficients))

    def test_ill_conditioned_matrix(self):
        """Test with ill-conditioned matrix (should trigger QR)."""
        # Create ill-conditioned matrix
        X_ill = np.array([[1, 1.0000000001], [2, 2.0000000002], [3, 3.0000000003]])
        y_ill = np.array([1, 2, 3])

        coefficients = self.ols.multivariate_ols(X_ill, y_ill)
        assert coefficients is not None
        assert all(np.isfinite(coefficients))

    def test_constant_y_values(self):
        """Test with constant y values."""
        X_normal = np.array([[1, 2], [3, 4], [5, 6]])
        y_constant = np.array([5, 5, 5])  # All same value

        coefficients = self.ols.multivariate_ols(X_normal, y_constant)
        assert coefficients is not None

        # Should predict constant value
        X_with_intercept = np.column_stack([np.ones(len(y_constant)), X_normal])
        predictions = X_with_intercept @ coefficients
        np.testing.assert_allclose(predictions, y_constant, atol=1e-10)

# pylint: disable=attribute-defined-outside-init
class TestPolynomialRegression:
    """Test cases for PolynomialRegression class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.poly_reg = PolynomialRegression(degree=2)

        # Test data for quadratic: y = 1 + 2x + 3x^2
        self.x_quad = np.array([1, 2, 3, 4, 5])
        self.y_quad = 1 + 2*self.x_quad + 3*self.x_quad**2

    def test_polynomial_regression_initialization(self):
        """Test PolynomialRegression initialization."""
        poly_reg = PolynomialRegression(degree=3)
        assert poly_reg.degree == 3
        assert poly_reg.coefficients is None

    def test_generate_polynomial_features(self):
        """Test polynomial feature generation."""
        x = np.array([1, 2, 3])
        # pylint: disable=protected-access
        features = self.poly_reg._generate_polynomial_features(x)

        # For degree=2, should get [x, x^2]
        expected = np.array([[1, 1], [2, 4], [3, 9]])
        np.testing.assert_array_equal(features, expected)

    def test_fit_method(self):
        """Test fit method."""
        result = self.poly_reg.fit(self.x_quad, self.y_quad)

        # Should return self
        assert result is self.poly_reg

        # Should set coefficients
        assert self.poly_reg.coefficients is not None
        assert len(self.poly_reg.coefficients) == 3  # intercept + degree terms
