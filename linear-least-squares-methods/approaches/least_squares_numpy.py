"""LeastSquares implementation in Python with numpy."""

import numpy as np
from sklearn.linear_model import lasso_path, enet_path

# Global constants used for bold text and red warning messages.
S_BOLD = "\033[1m"
E_BOLD = "\033[0m"
S_RED = "\033[91m"
E_RED = "\033[0m"

def fit_error_handling(coefficients):
    """fit error handling for Regression model."""
    if coefficients is None:
        raise ValueError("Model not fitted yet. Call fit() first.")

class LeastSquares:
    """LeastSquares implementation just with Numpy library."""

    def __init__(self, type_regression="PolynomialRegression"):
        self.type_regression=type_regression

        types_of_regression = ["PolynomialRegression", "RidgeRegression", "LassoRegression", "ElasticNetRegression"]
        count = 0
        for i, type_name in enumerate(types_of_regression):
            if type_regression == type_name:
                count += 1
                break
            if i == len(types_of_regression) and count == 0:
                raise ValueError(f"Type {self.type_regression} is not a valid predefined type.")

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

        # Threshold for switching, based on the condition number of the matrix, from using normal equations to QR decomposition.
        treshold_for_QR_decomposition = 1e6
        cond_number = 0

        if self.type_regression == "RidgeRegression":
            def multivariate_ols_ridge(X):
                """Modified Ridge Regression implementation."""
                XtX = X.T @ X
                regularization = self.alpha * np.eye(X.shape[1])
                regularization[0, 0] = 0
                XtX_modified = XtX + regularization
                cond_number = np.linalg.cond(XtX_modified)
                return cond_number
            cond_number = multivariate_ols_ridge(X)

        elif self.type_regression == "LassoRegression":
            return self._coordinate_descent_lasso(X, Y)

        elif self.type_regression == "ElasticNetRegression":
            return self._coordinate_descent_elasticnet(X, Y)

        elif self.type_regression == "PolynomialRegression":
            def multivariate_ols_standard(X):
                """ Standard LeastSquares for Standard PolynomialRegression """
                cond_number = np.linalg.cond(X.T @ X)
                return cond_number
            cond_number = multivariate_ols_standard(X)

        if cond_number <= 0:
            raise ValueError("Condition number is not larger than 0")
        if cond_number >= 1e13 and cond_number <= 1e15:
            print(f"\n! {S_RED}Warning:{E_RED} Matrix X is singular or extremely poorly conditioned {S_RED}!{E_RED}\n")
        elif cond_number >= 1e15:
            print(f"\n! {S_RED}Warning:{E_RED} Matrix X is singular or extremely poorly conditioned {S_RED}!{E_RED}")
            print(f"Condition number: {cond_number}\n")
            print("Recommendation: QR decomposition is probably extremely unstable; for better numerical stability, consider using SVD decomposition, though it may be slower.\n")

        if cond_number < 1e15:
            print(f"Condition number (in the 2-norm) for matrix X*X^T is: {cond_number}\n")

        if cond_number < treshold_for_QR_decomposition:
            w = self.normal_equations(X, Y)
        else:
            w = self.qr_decomposition(X, Y)
        return w

    def normal_equations(self, X, Y):
        """Compute LeastSquares coefficients using normal equations method."""

        if self.type_regression == "RidgeRegression":
            def normal_equations_ridge(X):
                """Compute Ridge regularized X^T X matrix with L2 penalty."""
                XtX = X.T @ X
                regularization = self.alpha * np.eye(X.shape[1])
                regularization[0, 0] = 0
                XtX_ridge = XtX + regularization
                return XtX_ridge
            XtX = normal_equations_ridge(X)


        else: # type_regression="PolynomialRegression"
            def normal_equations_standard(X, Y):
                """Compute standard X^T X and X^T Y matrices for OLS."""
                XtY = X.T @ Y
                XtX = X.T @ X
                return XtX, XtY
            XtX, XtY = normal_equations_standard(X, Y)

        try:
            w = np.linalg.solve(XtX, X.T @ Y)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(XtX) @ X.T @ Y
        return w

    def qr_decomposition(self, X, Y):
        """Compute LeastSquares coefficients using QR decomposition method for better numerical stability."""

        if self.type_regression == "RidgeRegression":
            def qr_decomposition_ridge(X, Y):
                """Perform QR decomposition on Ridge-extended system [X; sqrt(α)I]."""
                n_features = X.shape[1]
                sqrt_alpha = np.sqrt(self.alpha)

                I_reg = np.eye(n_features)
                I_reg[0, 0] = 0
                X_extended = np.vstack([X, sqrt_alpha * I_reg])
                Y_extended = np.hstack([Y, np.zeros(n_features)])

                Q, R = np.linalg.qr(X_extended)
                QtY = Q.T @ Y_extended
                return Q, R, QtY
            Q, R, QtY = qr_decomposition_ridge(X, Y)

        else: # type_regression="PolynomialRegression"
            def qr_decomposition_standard(X, Y):
                """Perform standard QR decomposition on matrix X."""
                Q, R = np.linalg.qr(X)
                QtY = Q.T @ Y
                return Q, R, QtY
            Q, R, QtY = qr_decomposition_standard(X, Y)

        try:
            w = np.linalg.solve(R, QtY)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(R) @ QtY
        return w


class CoordinateDescent(LeastSquares):
    """Coordinate descent algorithms for L1/L2 regularized regression inheriting from LeastSquares."""

    def _coordinate_descent_lasso(self, X, Y):
        """Lasso coordinate descent implementation using sklearn."""
        X_features = X[:, 1:]
        X_centered = X_features - np.mean(X_features, axis=0)
        y_centered = Y - np.mean(Y)
        
        # sklearn coordinate descent
        coefficients_features, dual_gap, _, n_iter = lasso_path(
            X_centered, y_centered,
            alphas=[self.alpha],
            max_iter=self.max_iter,
            tol=self.tol,
            return_n_iter=True
        )
        
        print(f"Lasso via CoordinateDescent: Converged in {n_iter[0]} iterations, dual gap: {dual_gap[0]:.2e}")
        
        # Reconstruct intercept: LeastSquares method
        feature_coeffs = coefficients_features[:, 0]
        intercept = np.mean(Y) - np.mean(X_features, axis=0) @ feature_coeffs
        
        # format: [intercept, features]
        return np.concatenate([[intercept], feature_coeffs])

    def _coordinate_descent_elasticnet(self, X, Y):
        """ElasticNet coordinate descent implementation using sklearn."""
        X_features = X[:, 1:]  # Without intercept column
        X_centered = X_features - np.mean(X_features, axis=0)
        y_centered = Y - np.mean(Y)
        
        # sklearn elastic net coordinate descent
        coefficients_features, dual_gap, _, n_iter = enet_path(
            X_centered, y_centered,
            l1_ratio=self.l1_ratio,
            alphas=[self.alpha],
            max_iter=self.max_iter,
            tol=self.tol,
            return_n_iter=True
        )
        
        print(f"ElasticNet via CoordinateDescent: Converged in {n_iter[0]} iterations, dual gap: {dual_gap[0]:.2e}")
        
        # Reconstruct intercept (LeastSquares method)
        feature_coeffs = coefficients_features[:, 0]
        intercept = np.mean(Y) - np.mean(X_features, axis=0) @ feature_coeffs
        
        # Return in LeastSquares format [intercept, features]
        return np.concatenate([[intercept], feature_coeffs])


class PolynomialRegression(LeastSquares):
    """Standard Polynomial regression using LeastSquares"""

    def __init__(self, degree, type_regression="PolynomialRegression"):
        super().__init__(type_regression="PolynomialRegression")
        self.degree = degree
        self.coefficients = None

    def fit(self, x, y):
        """Fit polynomial regression model."""
        X_polynomial = self._generate_polynomial_features(x)
        self.coefficients = self.multivariate_ols(X_polynomial, y)
        return self

    def predict(self, x):
        """Predict using fitted polynomial model."""
        fit_error_handling(self.coefficients)
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

class RidgeRegression(LeastSquares):
    """Ridge regression using LeastSquares infrastructure"""

    def __init__(self, alpha=1.0):
        super().__init__(type_regression="RidgeRegression", alpha=alpha)

    def fit(self, x, y):
        """Fit Ridge regression model."""
        X_with_intercept = np.column_stack([np.ones(len(y)), x])
        self.coefficients = self.multivariate_ols(X_with_intercept, y)
        return self

    def predict(self, x):
        """Prediction using Ridge coefficients."""
        fit_error_handling(self.coefficients)
        X_with_intercept = np.column_stack([np.ones(len(x)), x])
        return X_with_intercept @ self.coefficients


class LassoRegression(CoordinateDescent):
    """Lasso regression using CoordinateDescent infrastructure"""

    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        super().__init__(type_regression="LassoRegression", alpha=alpha)
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, x, y):
        """Fit Lasso regression model."""
        X_with_intercept = np.column_stack([np.ones(len(y)), x])
        self.coefficients = self.multivariate_ols(X_with_intercept, y)
        return self

    def predict(self, x):
        """Prediction using Lasso coefficients."""
        fit_error_handling(self.coefficients)
        X_with_intercept = np.column_stack([np.ones(len(x)), x])
        return X_with_intercept @ self.coefficients


class ElasticNetRegression(CoordinateDescent):
    """Elastic Net regression using CoordinateDescent infrastructure"""

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        super().__init__(type_regression="ElasticNetRegression", alpha=alpha)
        self.l1_ratio = l1_ratio  # L1 vs L2 mix: 0=Ridge, 1=Lasso, 0.5=Equal mix
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, x, y):
        """Fit ElasticNet regression model."""
        X_with_intercept = np.column_stack([np.ones(len(y)), x])
        self.coefficients = self.multivariate_ols(X_with_intercept, y)
        return self

    def predict(self, x):
        """Prediction using ElasticNet coefficients."""
        fit_error_handling(self.coefficients)
        X_with_intercept = np.column_stack([np.ones(len(x)), x])
        return X_with_intercept @ self.coefficients
