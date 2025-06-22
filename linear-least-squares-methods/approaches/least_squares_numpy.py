"""LeastSquares implementation in Python with numpy."""

import numpy as np
from sklearn.linear_model import Lasso, ElasticNet

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
        self.type_regression = type_regression
        self.alpha = 1.0  # Default alpha for Ridge/Lasso/ElasticNet
        self.l1_ratio = 0.5  # Default l1_ratio for ElasticNet
        self.max_iter = 1000
        self.tol = 1e-4

        types_of_regression = ["PolynomialRegression", "RidgeRegression", "LassoRegression", "ElasticNetRegression"]
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
        # Přidáme sloupec jedniček pro intercept
        X = np.column_stack([np.ones(len(Y)), X])

        n_rows, n_cols = X.shape
        if n_rows < n_cols:
            raise ValueError(f"Matrix X must have more rows than columns. Got {n_rows} rows and {n_cols} columns")

        # Threshold for switching from normal equations to QR decomposition
        threshold_for_QR_decomposition = 1e6
        cond_number = 0

        if self.type_regression == "RidgeRegression":
            # Vypočítáme condition number pro Ridge
            XtX = X.T @ X
            regularization = self.alpha * np.eye(X.shape[1])
            regularization[0, 0] = 0  # Neregularizujeme intercept
            XtX_modified = XtX + regularization
            cond_number = np.linalg.cond(XtX_modified)

        elif self.type_regression == "LassoRegression":
            return self._coordinate_descent_lasso(X, Y)

        elif self.type_regression == "ElasticNetRegression":
            return self._coordinate_descent_elasticnet(X, Y)

        elif self.type_regression == "PolynomialRegression":
            # Standard LeastSquares
            cond_number = np.linalg.cond(X.T @ X)

        # Kontrola condition number
        if cond_number <= 0:
            raise ValueError("Condition number is not larger than 0")

        if cond_number >= 1e13 and cond_number < 1e15:
            print(f"\n! {S_RED}Warning:{E_RED} Matrix X is poorly conditioned {S_RED}!{E_RED}")
            print(f"Condition number: {cond_number:.2e}\n")
        elif cond_number >= 1e15:
            print(f"\n! {S_RED}Warning:{E_RED} Matrix X is singular or extremely poorly conditioned {S_RED}!{E_RED}")
            print(f"Condition number: {cond_number:.2e}")
            print("Using QR decomposition for better numerical stability.\n")

        if cond_number < threshold_for_QR_decomposition:
            w = self.normal_equations(X, Y)
        else:
            print(f"Condition number: {cond_number:.2e} - using QR decomposition")
            w = self.qr_decomposition(X, Y)

        return w

    def normal_equations(self, X, Y):
        """Compute LeastSquares coefficients using normal equations method."""
        XtY = X.T @ Y

        if self.type_regression == "RidgeRegression":
            XtX = X.T @ X
            regularization = self.alpha * np.eye(X.shape[1])
            regularization[0, 0] = 0
            XtX_ridge = XtX + regularization
            try:
                w = np.linalg.solve(XtX_ridge, XtY)
            except np.linalg.LinAlgError:
                w = np.linalg.pinv(XtX_ridge) @ XtY
        else:
            XtX = X.T @ X
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
        X_features = X[:, 1:]  # Bez interceptu

        # Použijeme přímo Lasso model
        lasso = Lasso(alpha=self.alpha, max_iter=self.max_iter,
                      tol=self.tol, fit_intercept=True)
        lasso.fit(X_features, Y)

        print(f"Lasso via CoordinateDescent: Converged in {lasso.n_iter_} iterations")

        # Vrátíme koeficienty včetně interceptu
        return np.concatenate([[lasso.intercept_], lasso.coef_])

    def _coordinate_descent_elasticnet(self, X, Y):
        """ElasticNet coordinate descent implementation using sklearn."""
        X_features = X[:, 1:]  # Bez interceptu

        # Použijeme přímo ElasticNet model
        enet = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                          max_iter=self.max_iter, tol=self.tol, fit_intercept=True)
        enet.fit(X_features, Y)

        print(f"ElasticNet via CoordinateDescent: Converged in {enet.n_iter_} iterations")

        # Vrátíme koeficienty včetně interceptu
        return np.concatenate([[enet.intercept_], enet.coef_])


class PolynomialRegression(LeastSquares):
    """Standard Polynomial regression using LeastSquares"""

    def __init__(self, degree, type_regression="PolynomialRegression", normalize=True):
        super().__init__(type_regression="PolynomialRegression")
        self.degree = degree
        self.coefficients = None
        self.normalize = normalize
        self.x_min = None
        self.x_max = None

    def fit(self, x, y):
        """Fit polynomial regression model."""
        x = np.array(x)
        y = np.array(y)

        if x.ndim > 1:
            x = x.flatten()

        # Uložíme rozsah pro pozdější predikci
        self.x_min = x.min()
        self.x_max = x.max()

        X_polynomial = self._generate_polynomial_features(x)
        self.coefficients = self.multivariate_ols(X_polynomial, y)
        return self

    def predict(self, x):
        """Predict using fitted polynomial model."""
        fit_error_handling(self.coefficients)
        x = np.array(x)

        if x.ndim > 1:
            x = x.flatten()

        X_polynomial = self._generate_polynomial_features(x)
        X_polynomial_with_intercept = np.column_stack([np.ones(len(x)), X_polynomial])
        return X_polynomial_with_intercept @ self.coefficients

    def _generate_polynomial_features(self, x):
        """Generate polynomial features with optional normalization."""
        x = np.array(x).flatten()

        if self.normalize and self.degree > 3:
            # Normalizace x do intervalu [-1, 1] pro lepší numerickou stabilitu
            if self.x_max - self.x_min > 1e-10:
                x_normalized = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
            else:
                x_normalized = x
        else:
            x_normalized = x

        polynomial_features = []
        for i in range(1, self.degree + 1):
            polynomial_features.append(x_normalized ** i)

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

        # Pokud x je 1D, uděláme z něj 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Zavoláme multivariate_ols která přidá intercept
        self.coefficients = self.multivariate_ols(x, y)
        return self

    def predict(self, x):
        """Prediction using Ridge coefficients."""
        fit_error_handling(self.coefficients)
        x = np.array(x)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        X_with_intercept = np.column_stack([np.ones(len(x)), x])
        return X_with_intercept @ self.coefficients


class LassoRegression(LeastSquares):
    """Lasso regression using CoordinateDescent infrastructure"""

    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        super().__init__(type_regression="LassoRegression")
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients = None

    def fit(self, x, y):
        """Fit Lasso regression model."""
        x = np.array(x)
        y = np.array(y)

        # Pokud x je 1D, uděláme z něj 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Zavoláme multivariate_ols která přidá intercept
        self.coefficients = self.multivariate_ols(x, y)
        return self

    def predict(self, x):
        """Prediction using Lasso coefficients."""
        fit_error_handling(self.coefficients)
        x = np.array(x)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        X_with_intercept = np.column_stack([np.ones(len(x)), x])
        return X_with_intercept @ self.coefficients


class ElasticNetRegression(LeastSquares):
    """Elastic Net regression using CoordinateDescent infrastructure"""

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        super().__init__(type_regression="ElasticNetRegression")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients = None

    def fit(self, x, y):
        """Fit ElasticNet regression model."""
        x = np.array(x)
        y = np.array(y)

        # Pokud x je 1D, uděláme z něj 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Zavoláme multivariate_ols která přidá intercept
        self.coefficients = self.multivariate_ols(x, y)
        return self

    def predict(self, x):
        """Prediction using ElasticNet coefficients."""
        fit_error_handling(self.coefficients)
        x = np.array(x)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        X_with_intercept = np.column_stack([np.ones(len(x)), x])
        return X_with_intercept @ self.coefficients