"""LeastSquares implementation in Python with numpy - FIXED VERSION."""

import warnings
import numpy as np
from sklearn.linear_model import Lasso, ElasticNet

from constants import S_RED, E_RED

warnings.filterwarnings('ignore', message='Objective did not converge')


def generate_polynomial_features(X, degree):
    """Generate polynomial features up to specified degree with stable scaling."""
    X_flat = X.flatten() if X.ndim > 1 else X
    # Always normalize to [0, 1] range for better numerical stability
    x_min, x_max = X_flat.min(), X_flat.max()
    if x_max - x_min > 1e-10:
        # Normalize to [0, 1] range instead of [-1, 1]
        X_normalized = (X_flat - x_min) / (x_max - x_min)
    else:
        X_normalized = X_flat
    polynomial_features = []
    for d in range(1, degree + 1):
        feature = X_normalized ** d
        polynomial_features.append(feature)
    return np.column_stack(polynomial_features)


def fit_error_handling(coefficients):
    """fit error handling for Regression model."""
    if coefficients is None:
        raise ValueError("Model not fitted yet. Call fit() first.")


class LeastSquares:
    """LeastSquares implementation just with Numpy library."""

    def __init__(self, type_regression="LinearRegression"):
        self.type_regression = type_regression
        self.alpha = 1.0  # Default alpha for Ridge/Lasso/ElasticNet
        self.l1_ratio = 0.5  # Default l1_ratio for ElasticNet
        self.max_iter = 50000
        self.tol = 1e-4
        self.condition_number = None  # Store condition number for printing

        types_of_regression = ["LinearRegression", "RidgeRegression", "LassoRegression", "ElasticNetRegression"]
        if type_regression not in types_of_regression:
            raise ValueError(f"Type {self.type_regression} is not a valid predefined type.")

    def multivariate_ols(self, X, Y):
        """
        X: feature matrix
            Format: (N, p), where N >= p and N = number of observations, p = number of features
        Y: target variable vector
            Format: (N, ) - one-dimensional vector
        return: w = coefficient vector [w₀, w₁, w₂, ..., wₚ]
        """
        # Add column of ones for intercept
        X = np.column_stack([np.ones(len(Y)), X])

        n_rows, n_cols = X.shape
        if n_rows < n_cols:
            raise ValueError(f"Matrix X must have more rows than columns. Got {n_rows} rows and {n_cols} columns")

        # Threshold for switching from normal equations to QR decomposition
        threshold_for_QR_decomposition = 1e6
        cond_number = 0

        if self.type_regression == "RidgeRegression":
            cond_number = self._calculate_ridge_condition_number(X)
            self.condition_number = cond_number

        elif self.type_regression == "LassoRegression":
            self.condition_number = None  # Lasso doesn't use direct condition number calculation
            return self._coordinate_descent_lasso(X, Y)

        elif self.type_regression == "ElasticNetRegression":
            self.condition_number = None  # ElasticNet doesn't use direct condition number calculation
            return self._coordinate_descent_elasticnet(X, Y)

        elif self.type_regression == "LinearRegression":
            cond_number = self._calculate_standard_condition_number(X)
            self.condition_number = cond_number

        # Check condition number
        if cond_number <= 0:
            cond_number = 1e20  # Force QR decomposition

        if 1e13 <= cond_number < 1e15:
            print(f"\n! {S_RED}Warning:{E_RED} Matrix X is poorly conditioned {S_RED}!{E_RED}")
            print(f"Condition number: {cond_number:.2e}\n")
        elif cond_number >= 1e15:
            print(f"\n{S_RED}Warning:{E_RED} Matrix X is singular or extremely poorly conditioned")
            print(f"Condition number: {cond_number:.2e}")
            print("Using QR decomposition for better numerical stability.\n")

        if cond_number < threshold_for_QR_decomposition:
            w = self.normal_equations(X, Y)
        else:
            # Store but don't automatically print condition number
            # print(f"Condition number: {cond_number:.2e} - using QR decomposition")
            w = self.qr_decomposition(X, Y)

        return w

    def normal_equations(self, X, Y):
        """Compute LeastSquares coefficients using normal equations method."""
        XtY = X.T @ Y

        if self.type_regression == "RidgeRegression":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                XtX = X.T @ X
                regularization = self.alpha * np.eye(X.shape[1])
                regularization[0, 0] = 0
                XtX_ridge = XtX + regularization

                # Check for numerical issues
                if np.any(np.isnan(XtX_ridge)) or np.any(np.isinf(XtX_ridge)):
                    # Use QR decomposition as fallback
                    return self.qr_decomposition(X, Y)

                try:
                    w = np.linalg.solve(XtX_ridge, XtY)
                except np.linalg.LinAlgError:
                    w = np.linalg.pinv(XtX_ridge) @ XtY
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                XtX = X.T @ X

                # Check for numerical issues
                if np.any(np.isnan(XtX)) or np.any(np.isinf(XtX)):
                    # Use QR decomposition as fallback
                    return self.qr_decomposition(X, Y)

                try:
                    w = np.linalg.solve(XtX, XtY)
                except np.linalg.LinAlgError:
                    w = np.linalg.pinv(XtX) @ XtY

        return w

    def qr_decomposition(self, X, Y):
        """Compute LeastSquares coefficients using QR decomposition method."""
        if self.type_regression == "RidgeRegression":
            n_features = X.shape[1]
            sqrt_alpha = np.sqrt(self.alpha)

            I_reg = np.eye(n_features)
            I_reg[0, 0] = 0
            X_extended = np.vstack([X, sqrt_alpha * I_reg])
            Y_extended = np.hstack([Y, np.zeros(n_features)])

            Q, R = np.linalg.qr(X_extended)
            QtY = Q.T @ Y_extended
        else:
            Q, R = np.linalg.qr(X)
            QtY = Q.T @ Y

        try:
            w = np.linalg.solve(R, QtY)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(R) @ QtY

        return w

    def _coordinate_descent_lasso(self, X, Y):
        """Lasso coordinate descent implementation using sklearn."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
        X_features = X[:, 1:]  # Without intercept

        # Use Lasso model directly
        lasso = Lasso(alpha=self.alpha, max_iter=self.max_iter,
                      tol=self.tol, fit_intercept=True)
        lasso.fit(X_features, Y)

        # Return coefficients including intercept
        return np.concatenate([[lasso.intercept_], lasso.coef_])

    def _coordinate_descent_elasticnet(self, X, Y):
        """ElasticNet coordinate descent implementation using sklearn."""
        X_features = X[:, 1:]  # Without intercept

        # Use ElasticNet model directly
        enet = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                          max_iter=self.max_iter, tol=self.tol, fit_intercept=True)
        enet.fit(X_features, Y)

        # Return coefficients including intercept
        return np.concatenate([[enet.intercept_], enet.coef_])

    def _calculate_ridge_condition_number(self, X):
        """Calculate condition number for Ridge regression."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            XtX = X.T @ X
            regularization = self.alpha * np.eye(X.shape[1])
            regularization[0, 0] = 0  # Don't regularize intercept
            XtX_modified = XtX + regularization

            # Check for numerical issues
            if np.any(np.isnan(XtX_modified)) or np.any(np.isinf(XtX_modified)):
                return 1e20  # Force QR decomposition
            return np.linalg.cond(XtX_modified)

    def _calculate_standard_condition_number(self, X):
        """Calculate condition number for standard regression."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            XtX = X.T @ X

            # Check for numerical issues
            if np.any(np.isnan(XtX)) or np.any(np.isinf(XtX)):
                return 1e20  # Force QR decomposition

            try:
                return np.linalg.cond(XtX)
            except (np.linalg.LinAlgError, ValueError, RuntimeError):
                return 1e20  # Force QR decomposition


class LinearRegression(LeastSquares):
    """Standard Polynomial regression using LeastSquares - FIXED VERSION"""

    def __init__(self, degree, type_regression="LinearRegression", normalize=True):
        super().__init__(type_regression="LinearRegression")
        self.degree = degree
        self.coefficients = None
        self.normalize = normalize
        self.x_min = None
        self.x_max = None

    def fit(self, x, y):
        """Fit polynomial regression model with better numerical stability."""
        x = np.array(x)
        y = np.array(y)

        if x.ndim > 1:
            x = x.flatten()

        # Save range for later prediction
        self.x_min = x.min()
        self.x_max = x.max()

        X_polynomial = self._generate_polynomial_features(x)

        # Add small regularization for very high degree polynomials
        if self.degree > 5:
            self.alpha = 1e-8
            self.type_regression = "RidgeRegression"

        self.coefficients = self.multivariate_ols(X_polynomial, y)

        # Reset to original type
        self.type_regression = "LinearRegression"

        return self

    def predict(self, x):
        """Predict using fitted polynomial model."""
        fit_error_handling(self.coefficients)
        x = np.array(x)

        # Check if we already have polynomial features (from visualization)
        if x.ndim == 2 and x.shape[1] == self.degree:
            # Already polynomial features, just add intercept
            X_polynomial_with_intercept = np.column_stack([np.ones(x.shape[0]), x])
        else:
            # Generate polynomial features from raw x values
            if x.ndim > 1:
                x = x.flatten()
            X_polynomial = self._generate_polynomial_features(x)
            X_polynomial_with_intercept = np.column_stack([np.ones(len(x)), X_polynomial])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            predictions = X_polynomial_with_intercept @ self.coefficients

            # Handle any NaN or inf values
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                # Return zeros or original y mean as fallback
                predictions = np.zeros_like(predictions)

        return predictions

    def _generate_polynomial_features(self, x):
        """Generate polynomial features with enhanced normalization for stability."""
        x = np.array(x).flatten()

        if self.normalize and self.degree > 3:
            # Normalize x to interval [-1, 1] for better numerical stability
            if self.x_max - self.x_min > 1e-10:
                x_normalized = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
            else:
                x_normalized = x
        else:
            x_normalized = x

        polynomial_features = []
        for i in range(1, self.degree + 1):
            # For very high degrees, scale down the features
            if i > 5:
                # Use additional scaling to prevent overflow
                feature = x_normalized ** i / (10 ** (i - 5))
            else:
                feature = x_normalized ** i

            # Check for numerical issues
            if np.any(np.isnan(feature)) or np.any(np.isinf(feature)):
                # Replace with scaled version
                feature = np.sign(x_normalized) * (np.abs(x_normalized) ** (i / 2))

            polynomial_features.append(feature)

        return np.column_stack(polynomial_features)


class RidgeRegression(LeastSquares):
    """Ridge regression using LeastSquares infrastructure"""

    def __init__(self, alpha=1.0):
        super().__init__(type_regression="RidgeRegression")
        self.alpha = alpha
        self.coefficients = None

    def fit(self, x, y):
        """Fit Ridge regression model."""
        x = np.array(x)
        y = np.array(y)

        # If x is 1D, make it 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Call multivariate_ols which adds intercept
        self.coefficients = self.multivariate_ols(x, y)
        return self

    def predict(self, x):
        """Prediction using Ridge coefficients."""
        fit_error_handling(self.coefficients)
        x = np.array(x)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Check if this is polynomial regression (has degree attribute)
        if hasattr(self, 'degree') and self.degree > 1:
            # Check if we already have polynomial features (from visualization)
            if x.shape[1] == self.degree:
                # Already polynomial features, just add intercept
                X_with_intercept = np.column_stack([np.ones(x.shape[0]), x])
            else:
                # Generate polynomial features for prediction
                X_poly = generate_polynomial_features(x, self.degree)
                X_with_intercept = np.column_stack([np.ones(x.shape[0]), X_poly])
        else:
            # Regular linear features
            X_with_intercept = np.column_stack([np.ones(len(x)), x])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            predictions = X_with_intercept @ self.coefficients

            # Handle any NaN or inf values
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                predictions = np.zeros_like(predictions)

        return predictions


class LassoRegression(LeastSquares):
    """Lasso regression using CoordinateDescent infrastructure"""

    def __init__(self, alpha=1.0, max_iter=50000, tol=1e-4):
        super().__init__(type_regression="LassoRegression")
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients = None
        self.model = None  # For sklearn models

    def fit(self, x, y):
        """Fit Lasso regression model using sklearn Lasso."""
        x = np.array(x)
        y = np.array(y)

        # If x is 1D, make it 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Use sklearn Lasso directly for proper L1 regularization
        lasso = Lasso(alpha=self.alpha, max_iter=self.max_iter,
                      tol=self.tol, fit_intercept=True)
        lasso.fit(x, y)

        # Store sklearn model for prediction
        self.model = lasso
        # Store coefficients in our format [intercept, coef1, coef2, ...]
        self.coefficients = np.concatenate([[lasso.intercept_], lasso.coef_])
        return self

    def predict(self, x):
        """Prediction using sklearn Lasso model."""
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Check if this is polynomial regression (has degree attribute)
        if hasattr(self, 'degree') and self.degree > 1:
            # Check if we already have polynomial features (from visualization)
            if x.shape[1] != self.degree:
                x = generate_polynomial_features(x, self.degree)

        return self.model.predict(x)


class ElasticNetRegression(LeastSquares):
    """Elastic Net regression using CoordinateDescent infrastructure"""

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=50000, tol=1e-4):
        super().__init__(type_regression="ElasticNetRegression")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients = None
        self.model = None  # For sklearn models

    def fit(self, x, y):
        """Fit ElasticNet regression model using sklearn ElasticNet."""
        x = np.array(x)
        y = np.array(y)

        # If x is 1D, make it 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Use sklearn ElasticNet directly for proper L1+L2 regularization
        elasticnet = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                               max_iter=self.max_iter, tol=self.tol, fit_intercept=True)
        elasticnet.fit(x, y)

        # Store sklearn model for prediction
        self.model = elasticnet
        # Store coefficients in our format [intercept, coef1, coef2, ...]
        self.coefficients = np.concatenate([[elasticnet.intercept_], elasticnet.coef_])
        return self

    def predict(self, x):
        """Prediction using sklearn ElasticNet model."""
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Check if this is polynomial regression (has degree attribute)
        if hasattr(self, 'degree') and self.degree > 1:
            # Check if we already have polynomial features (from visualization)
            if x.shape[1] != self.degree:
                # Generate polynomial features for prediction
                x = generate_polynomial_features(x, self.degree)

        return self.model.predict(x)
