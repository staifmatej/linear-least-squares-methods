"""OLS implementation in Python with numpy."""
import numpy as np

class OLS:
    """OLS implementation just with Numpy library."""

    def __init__(self):
        pass

    def multivariate_ols(self, X, Y):
        """
            X: feature matrix
                Format: (N, p), where N >= p and N = number of observations, p = number of features
            Y: target variable vector
                Format: (N, ) - one-dimensional vector
            return: w = coefficient vector [w₀, w₁, w₂, ..., wₚ]
        """

        X = np.column_stack([np.ones(len(Y)), X])

        n_rows, n_cols = X.shape
        if n_rows < n_cols:
            raise ValueError(f"Matrix X must have more rows than columns. Got {n_rows} rows and {n_cols} columns")
        cond_number = np.linalg.cond(X.T @ X)
        if cond_number < 1e6:
            w = self.normal_equations(X, Y)
        else:
            w = self.qr_decomposition(X, Y)

        return w

    def normal_equations(self, X, Y):
        """Compute OLS coefficients using normal equations method with singular matrix protection."""
        XtY =  X.T @ Y
        XtX = X.T @ X
        try:
            w = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(XtX) @ XtY
        return w

    def qr_decomposition(self, X, Y):
        """Compute OLS coefficients using QR decomposition method for better numerical stability."""
        Q, R = np.linalg.qr(X)
        QtY = Q.T @ Y
        try:
            w = np.linalg.solve(R, QtY)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(R) @ QtY
        return w

class PolynomialRegression(OLS):
    """Standard Polynomial regression using OLS"""

    def __init__(self, degree):
        self.degree = degree
        self.coefficients = None

    def fit(self, x, y):
        """Fit polynomial regression model."""
        X_polynomial = self._generate_polynomial_features(x)
        self.coefficients = self.multivariate_ols(X_polynomial, y)
        return self

    def predict(self, x):
        """Predict using fitted polynomial model."""
        X_polynomial = self._generate_polynomial_features(x)
        X_polynomial_with_intercept = np.column_stack([np.ones(len(x)), X_polynomial])
        return X_polynomial_with_intercept @ self.coefficients

    def _generate_polynomial_features(self, x):
        """Generate polynomial features up to specified degree."""
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        polynomial_features = []
        for i in range(1, self.degree + 1):
            polynomial_features.append(x**i)

        X_polynomial = np.column_stack(polynomial_features)

        return X_polynomial

class RidgeRegression(OLS):
    """Ridge regression using OLS"""

class LassoRegression(OLS):
    """Lasso regression using OLS"""

class ElasticNetRegression(OLS):
    """Elastic net regression using OLS"""
