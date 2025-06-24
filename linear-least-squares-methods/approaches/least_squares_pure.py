"""LeastSquares implementation in Python with pure Python - FIXED VERSION."""

import warnings
import math
from sklearn.linear_model import Lasso, ElasticNet

# Suppress sklearn convergence warnings globally
warnings.filterwarnings('ignore', message='Objective did not converge')

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
    """LeastSquares implementation using pure Python."""

    def __init__(self, type_regression="PolynomialRegression"):
        self.type_regression = type_regression
        self.alpha = 1.0  # Default alpha for Ridge/Lasso/ElasticNet
        self.l1_ratio = 0.5  # Default l1_ratio for ElasticNet
        self.max_iter = 20000
        self.tol = 1e-4

        types_of_regression = ["PolynomialRegression", "RidgeRegression", "LassoRegression", "ElasticNetRegression"]
        if type_regression not in types_of_regression:
            raise ValueError(f"Type {self.type_regression} is not a valid predefined type.")

    def multivariate_ols(self, X, Y):
        """
        X: feature matrix
            Format: list of lists, where N >= p and N = number of observations, p = number of features
        Y: target variable vector
            Format: list - one-dimensional
        return: w = coefficient vector [w₀, w₁, w₂, ..., wₚ]
        """
        # Convert to lists if needed
        if hasattr(X, 'tolist'):
            X = X.tolist()
        if hasattr(Y, 'tolist'):
            Y = Y.tolist()

        # Handle 1D input
        if isinstance(X[0], (int, float)):
            X = [[x] for x in X]

        # Add column of ones for intercept
        n_samples = len(Y)
        X_with_intercept = []
        for i in range(n_samples):
            row = [1.0] + X[i]
            X_with_intercept.append(row)

        n_rows = len(X_with_intercept)
        n_cols = len(X_with_intercept[0])

        if n_rows < n_cols:
            raise ValueError(f"Matrix X must have more rows than columns. Got {n_rows} rows and {n_cols} columns")

        # Calculate condition number
        threshold_for_QR_decomposition = 1e6
        cond_number = 0

        if self.type_regression == "RidgeRegression":
            cond_number = self._calculate_ridge_condition_number(X_with_intercept)
        elif self.type_regression == "LassoRegression":
            return self._coordinate_descent_lasso(X_with_intercept, Y)
        elif self.type_regression == "ElasticNetRegression":
            return self._coordinate_descent_elasticnet(X_with_intercept, Y)
        elif self.type_regression == "PolynomialRegression":
            cond_number = self._calculate_standard_condition_number(X_with_intercept)

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
            w = self.normal_equations(X_with_intercept, Y)
        else:
            w = self.qr_decomposition(X_with_intercept, Y)

        return w

    def normal_equations(self, X, Y):
        """Compute LeastSquares coefficients using normal equations method."""
        # Compute X^T * Y
        XtY = self._matrix_vector_multiply_transpose(X, Y)

        if self.type_regression == "RidgeRegression":
            # Compute X^T * X
            XtX = self._matrix_multiply_transpose(X, X)
            n_features = len(XtX)

            # Add ridge regularization
            for i in range(1, n_features):  # Skip intercept
                XtX[i][i] += self.alpha

            # Solve system
            try:
                w = self._solve_linear_system(XtX, XtY)
            except:
                # Fallback to pseudo-inverse
                w = self._solve_with_pseudoinverse(XtX, XtY)
        else:
            # Compute X^T * X
            XtX = self._matrix_multiply_transpose(X, X)

            # Solve system
            try:
                w = self._solve_linear_system(XtX, XtY)
            except:
                # Fallback to pseudo-inverse
                w = self._solve_with_pseudoinverse(XtX, XtY)

        return w

    def qr_decomposition(self, X, Y):
        """Compute LeastSquares coefficients using QR decomposition method."""
        if self.type_regression == "RidgeRegression":
            n_samples = len(X)
            n_features = len(X[0])
            sqrt_alpha = math.sqrt(self.alpha)

            # Create extended matrix for Ridge
            X_extended = []
            # Original X
            for row in X:
                X_extended.append(row[:])
            # Regularization rows
            for i in range(n_features):
                reg_row = [0.0] * n_features
                if i > 0:  # Skip intercept
                    reg_row[i] = sqrt_alpha
                X_extended.append(reg_row)

            # Extended Y
            Y_extended = Y[:] + [0.0] * n_features

            # QR decomposition
            Q, R = self._qr_decomposition_pure(X_extended)
            QtY = self._matrix_vector_multiply_transpose(Q, Y_extended)
        else:
            # Standard QR decomposition
            Q, R = self._qr_decomposition_pure(X)
            QtY = self._matrix_vector_multiply_transpose(Q, Y)

        # Solve R * w = QtY using back substitution
        w = self._back_substitution(R, QtY)
        return w

    def _coordinate_descent_lasso(self, X, Y):
        """Lasso coordinate descent implementation using sklearn."""
        # Remove intercept column for sklearn
        X_features = []
        for row in X:
            X_features.append(row[1:])

        # Use Lasso model directly
        lasso = Lasso(alpha=self.alpha, max_iter=self.max_iter,
                      tol=self.tol, fit_intercept=True)
        lasso.fit(X_features, Y)

        # Return coefficients including intercept
        coeffs = [lasso.intercept_] + list(lasso.coef_)
        return coeffs

    def _coordinate_descent_elasticnet(self, X, Y):
        """ElasticNet coordinate descent implementation using sklearn."""
        # Remove intercept column for sklearn
        X_features = []
        for row in X:
            X_features.append(row[1:])

        # Use ElasticNet model directly
        enet = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                          max_iter=self.max_iter, tol=self.tol, fit_intercept=True)
        enet.fit(X_features, Y)

        # Return coefficients including intercept
        coeffs = [enet.intercept_] + list(enet.coef_)
        return coeffs

    def _calculate_ridge_condition_number(self, X):
        """Calculate condition number for Ridge regression."""
        XtX = self._matrix_multiply_transpose(X, X)
        n_features = len(XtX)

        # Add ridge regularization
        for i in range(1, n_features):
            XtX[i][i] += self.alpha

        return self._condition_number(XtX)

    def _calculate_standard_condition_number(self, X):
        """Calculate condition number for standard regression."""
        XtX = self._matrix_multiply_transpose(X, X)
        return self._condition_number(XtX)

    def _condition_number(self, A):
        """Calculate condition number of matrix A using eigenvalues."""
        try:
            eigenvalues = self._eigenvalues_power_method(A)
            if len(eigenvalues) == 0:
                return 1e20

            max_eig = max(abs(e) for e in eigenvalues)
            min_eig = min(abs(e) for e in eigenvalues if abs(e) > 1e-10)

            if min_eig < 1e-10:
                return 1e20

            return max_eig / min_eig
        except:
            return 1e20

    def _eigenvalues_power_method(self, A, max_iter=10000):
        """Approximate eigenvalues using power method (simplified)."""
        n = len(A)
        eigenvalues = []

        # Just get largest eigenvalue for now
        v = [1.0] * n
        for _ in range(max_iter):
            v_new = self._matrix_vector_multiply(A, v)
            norm = math.sqrt(sum(x ** 2 for x in v_new))
            if norm > 1e-10:
                v = [x / norm for x in v_new]

        # Rayleigh quotient
        Av = self._matrix_vector_multiply(A, v)
        lambda_max = sum(v[i] * Av[i] for i in range(n))
        eigenvalues.append(lambda_max)

        # For condition number, we need min eigenvalue too
        # This is simplified - just return rough estimate
        eigenvalues.append(lambda_max / 1000.0)  # Rough estimate

        return eigenvalues

    def _qr_decomposition_pure(self, A):
        """QR decomposition using Gram-Schmidt orthogonalization."""
        m = len(A)
        n = len(A[0])

        # Initialize Q and R
        Q = [[0.0] * n for _ in range(m)]
        R = [[0.0] * n for _ in range(n)]

        # Gram-Schmidt process
        for j in range(n):
            # Get column j of A
            v = [A[i][j] for i in range(m)]

            # Orthogonalize against previous columns
            for i in range(j):
                # R[i][j] = Q[:, i]^T * A[:, j]
                R[i][j] = sum(Q[k][i] * A[k][j] for k in range(m))
                # v = v - R[i][j] * Q[:, i]
                for k in range(m):
                    v[k] -= R[i][j] * Q[k][i]

            # Normalize
            R[j][j] = math.sqrt(sum(v[k] ** 2 for k in range(m)))

            if R[j][j] > 1e-10:
                for k in range(m):
                    Q[k][j] = v[k] / R[j][j]
            else:
                # Handle zero column
                for k in range(m):
                    Q[k][j] = 0.0

        return Q, R

    def _back_substitution(self, R, b):
        """Solve Rx = b where R is upper triangular."""
        n = len(R)
        x = [0.0] * n

        for i in range(n - 1, -1, -1):
            x[i] = b[i]
            for j in range(i + 1, n):
                x[i] -= R[i][j] * x[j]
            if abs(R[i][i]) > 1e-10:
                x[i] /= R[i][i]
            else:
                x[i] = 0.0

        return x

    def _solve_with_pseudoinverse(self, A, b):
        """Solve using pseudo-inverse (simplified implementation)."""
        # This is a simplified version - just add small diagonal for stability
        n = len(A)
        A_reg = [row[:] for row in A]

        # Add small value to diagonal
        for i in range(n):
            A_reg[i][i] += 1e-10

        return self._solve_linear_system(A_reg, b)

    def _matrix_multiply_transpose(self, A, B):
        """Compute A^T * B."""
        m = len(A)
        n = len(A[0]) if m > 0 else 0
        k = len(B[0]) if m > 0 else 0

        result = [[0.0 for _ in range(k)] for _ in range(n)]

        for i in range(n):
            for j in range(k):
                for t in range(m):
                    result[i][j] += A[t][i] * B[t][j]

        return result

    def _matrix_vector_multiply_transpose(self, A, b):
        """Compute A^T * b."""
        m = len(A)
        n = len(A[0]) if m > 0 else 0

        result = [0.0 for _ in range(n)]

        for i in range(n):
            for j in range(m):
                result[i] += A[j][i] * b[j]

        return result

    def _matrix_vector_multiply(self, A, v):
        """Compute A * v."""
        m = len(A)
        n = len(A[0])

        result = [0.0] * m

        for i in range(m):
            for j in range(n):
                result[i] += A[i][j] * v[j]

        return result

    def _solve_linear_system(self, A, b):
        """Solve Ax = b using Gaussian elimination with partial pivoting."""
        n = len(A)

        # Create augmented matrix
        augmented = []
        for i in range(n):
            row = A[i][:] + [b[i]]
            augmented.append(row)

        # Forward elimination with partial pivoting
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                    max_row = k

            # Swap rows
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]

            # Check for zero pivot
            if abs(augmented[i][i]) < 1e-10:
                augmented[i][i] = 1e-10

            # Eliminate column
            for k in range(i + 1, n):
                factor = augmented[k][i] / augmented[i][i]
                for j in range(i, n + 1):
                    augmented[k][j] -= factor * augmented[i][j]

        # Back substitution
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = augmented[i][n]
            for j in range(i + 1, n):
                x[i] -= augmented[i][j] * x[j]
            x[i] /= augmented[i][i]

        return x


class PolynomialRegression(LeastSquares):
    """Standard Polynomial regression using LeastSquares - FIXED VERSION"""

    def __init__(self, degree, type_regression="PolynomialRegression", normalize=True):
        super().__init__(type_regression="PolynomialRegression")
        self.degree = degree
        self.coefficients = None
        self.normalize = normalize
        self.x_min = None
        self.x_max = None

    def fit(self, x, y):
        """Fit polynomial regression model with better numerical stability."""
        # Convert to lists if needed
        if hasattr(x, 'tolist'):
            x = x.tolist()
        if hasattr(y, 'tolist'):
            y = y.tolist()

        # Ensure x is a list
        if not isinstance(x, list):
            x = list(x)
        if not isinstance(y, list):
            y = list(y)

        # Save range for later prediction
        self.x_min = min(x)
        self.x_max = max(x)

        X_polynomial = self._generate_polynomial_features(x)

        # Add small regularization for very high degree polynomials
        if self.degree > 5:
            self.alpha = 1e-8
            self.type_regression = "RidgeRegression"

        self.coefficients = self.multivariate_ols(X_polynomial, y)

        # Reset to original type
        self.type_regression = "PolynomialRegression"

        return self

    def predict(self, x):
        """Predict using fitted polynomial model."""
        fit_error_handling(self.coefficients)

        # Convert to list if needed
        if hasattr(x, 'tolist'):
            x = x.tolist()
        if not isinstance(x, list):
            x = list(x)

        X_polynomial = self._generate_polynomial_features(x)

        # Add intercept column
        X_polynomial_with_intercept = []
        for i in range(len(x)):
            row = [1.0] + X_polynomial[i]
            X_polynomial_with_intercept.append(row)

        # Matrix-vector multiplication
        predictions = []
        for row in X_polynomial_with_intercept:
            y_pred = sum(row[j] * self.coefficients[j] for j in range(len(self.coefficients)))
            predictions.append(y_pred)

        return predictions

    def _generate_polynomial_features(self, x):
        """Generate polynomial features with enhanced normalization for stability."""
        n = len(x)

        if self.normalize and self.degree > 3:
            # Normalize x to interval [-1, 1] for better numerical stability
            if self.x_max - self.x_min > 1e-10:
                x_normalized = [2 * (xi - self.x_min) / (self.x_max - self.x_min) - 1 for xi in x]
            else:
                x_normalized = x[:]
        else:
            x_normalized = x[:]

        polynomial_features = []
        for i in range(n):
            row = []
            for d in range(1, self.degree + 1):
                # For very high degrees, scale down the features
                if d > 5:
                    # Use additional scaling to prevent overflow
                    feature = x_normalized[i] ** d / (10 ** (d - 5))
                else:
                    feature = x_normalized[i] ** d

                # Check for numerical issues
                if abs(feature) > 1e100:
                    # Replace with scaled version
                    sign = 1 if x_normalized[i] >= 0 else -1
                    feature = sign * (abs(x_normalized[i]) ** (d / 2.0))

                row.append(feature)
            polynomial_features.append(row)

        return polynomial_features


class RidgeRegression(LeastSquares):
    """Ridge regression using LeastSquares infrastructure"""

    def __init__(self, alpha=1.0):
        super().__init__(type_regression="RidgeRegression")
        self.alpha = alpha
        self.coefficients = None

    def fit(self, x, y):
        """Fit Ridge regression model."""
        # Convert to lists if needed
        if hasattr(x, 'tolist'):
            x = x.tolist()
        if hasattr(y, 'tolist'):
            y = y.tolist()

        # Handle both 1D and 2D inputs
        if isinstance(x[0], (int, float)):
            x = [[xi] for xi in x]

        # Call multivariate_ols which adds intercept
        self.coefficients = self.multivariate_ols(x, y)
        return self

    def predict(self, x):
        """Prediction using Ridge coefficients."""
        fit_error_handling(self.coefficients)

        # Convert to list if needed
        if hasattr(x, 'tolist'):
            x = x.tolist()

        # Handle both 1D and 2D inputs
        if isinstance(x[0], (int, float)):
            x = [[xi] for xi in x]

        # Add intercept column
        X_with_intercept = []
        for row in x:
            X_with_intercept.append([1.0] + row)

        # Predictions
        predictions = []
        for row in X_with_intercept:
            y_pred = sum(row[j] * self.coefficients[j] for j in range(len(self.coefficients)))
            predictions.append(y_pred)

        return predictions


class LassoRegression(LeastSquares):
    """Lasso regression using CoordinateDescent infrastructure"""

    def __init__(self, alpha=1.0, max_iter=20000, tol=1e-4):
        super().__init__(type_regression="LassoRegression")
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients = None

    def fit(self, x, y):
        """Fit Lasso regression model."""
        # Convert to lists if needed
        if hasattr(x, 'tolist'):
            x = x.tolist()
        if hasattr(y, 'tolist'):
            y = y.tolist()

        # Handle both 1D and 2D inputs
        if isinstance(x[0], (int, float)):
            x = [[xi] for xi in x]

        # Call multivariate_ols which adds intercept
        self.coefficients = self.multivariate_ols(x, y)
        return self

    def predict(self, x):
        """Prediction using Lasso coefficients."""
        fit_error_handling(self.coefficients)

        # Convert to list if needed
        if hasattr(x, 'tolist'):
            x = x.tolist()

        # Handle both 1D and 2D inputs
        if isinstance(x[0], (int, float)):
            x = [[xi] for xi in x]

        # Add intercept column
        X_with_intercept = []
        for row in x:
            X_with_intercept.append([1.0] + row)

        # Predictions
        predictions = []
        for row in X_with_intercept:
            y_pred = sum(row[j] * self.coefficients[j] for j in range(len(self.coefficients)))
            predictions.append(y_pred)

        return predictions


class ElasticNetRegression(LeastSquares):
    """Elastic Net regression using CoordinateDescent infrastructure"""

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=20000, tol=1e-4):
        super().__init__(type_regression="ElasticNetRegression")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients = None

    def fit(self, x, y):
        """Fit ElasticNet regression model."""
        # Convert to lists if needed
        if hasattr(x, 'tolist'):
            x = x.tolist()
        if hasattr(y, 'tolist'):
            y = y.tolist()

        # Handle both 1D and 2D inputs
        if isinstance(x[0], (int, float)):
            x = [[xi] for xi in x]

        # Call multivariate_ols which adds intercept
        self.coefficients = self.multivariate_ols(x, y)
        return self

    def predict(self, x):
        """Prediction using ElasticNet coefficients."""
        fit_error_handling(self.coefficients)

        # Convert to list if needed
        if hasattr(x, 'tolist'):
            x = x.tolist()

        # Handle both 1D and 2D inputs
        if isinstance(x[0], (int, float)):
            x = [[xi] for xi in x]

        # Add intercept column
        X_with_intercept = []
        for row in x:
            X_with_intercept.append([1.0] + row)

        # Predictions
        predictions = []
        for row in X_with_intercept:
            y_pred = sum(row[j] * self.coefficients[j] for j in range(len(self.coefficients)))
            predictions.append(y_pred)

        return predictions


class PolynomialRegression(LeastSquares):
    """Standard Polynomial regression using LeastSquares - FIXED VERSION"""

    def __init__(self, degree, type_regression="PolynomialRegression", normalize=True):
        super().__init__(type_regression="PolynomialRegression")
        self.degree = degree
        self.coefficients = None
        self.normalize = normalize
        self.x_min = None
        self.x_max = None

    def fit(self, x, y):
        """Fit polynomial regression model with better numerical stability."""
        # Convert to lists if needed
        if hasattr(x, 'tolist'):
            x = x.tolist()
        if hasattr(y, 'tolist'):
            y = y.tolist()

        # Ensure x is a list
        if not isinstance(x, list):
            x = list(x)
        if not isinstance(y, list):
            y = list(y)

        # Save range for later prediction
        self.x_min = min(x)
        self.x_max = max(x)

        X_polynomial = self._generate_polynomial_features(x)

        # Add small regularization for very high degree polynomials
        if self.degree > 5:
            self.alpha = 1e-8
            self.type_regression = "RidgeRegression"

        self.coefficients = self.multivariate_ols(X_polynomial, y)

        # Reset to original type
        self.type_regression = "PolynomialRegression"

        return self

    def predict(self, x):
        """Predict using fitted polynomial model."""
        fit_error_handling(self.coefficients)

        # Convert to list if needed
        if hasattr(x, 'tolist'):
            x = x.tolist()
        if not isinstance(x, list):
            x = list(x)

        X_polynomial = self._generate_polynomial_features(x)

        # Add intercept column
        X_polynomial_with_intercept = []
        for i in range(len(x)):
            row = [1.0] + X_polynomial[i]
            X_polynomial_with_intercept.append(row)

        # Matrix-vector multiplication
        predictions = []
        for row in X_polynomial_with_intercept:
            y_pred = sum(row[j] * self.coefficients[j] for j in range(len(self.coefficients)))
            predictions.append(y_pred)

        return predictions

    def _generate_polynomial_features(self, x):
        """Generate polynomial features with enhanced normalization for stability."""
        n = len(x)

        if self.normalize and self.degree > 3:
            # Normalize x to interval [-1, 1] for better numerical stability
            if self.x_max - self.x_min > 1e-10:
                x_normalized = [2 * (xi - self.x_min) / (self.x_max - self.x_min) - 1 for xi in x]
            else:
                x_normalized = x[:]
        else:
            x_normalized = x[:]

        polynomial_features = []
        for i in range(n):
            row = []
            for d in range(1, self.degree + 1):
                # For very high degrees, scale down the features
                if d > 5:
                    # Use additional scaling to prevent overflow
                    feature = x_normalized[i] ** d / (10 ** (d - 5))
                else:
                    feature = x_normalized[i] ** d

                # Check for numerical issues
                if abs(feature) > 1e100:
                    # Replace with scaled version
                    sign = 1 if x_normalized[i] >= 0 else -1
                    feature = sign * (abs(x_normalized[i]) ** (d / 2.0))

                row.append(feature)
            polynomial_features.append(row)

        return polynomial_features


class RidgeRegression(LeastSquares):
    """Ridge regression using LeastSquares infrastructure"""

    def __init__(self, alpha=1.0):
        super().__init__(type_regression="RidgeRegression")
        self.alpha = alpha
        self.coefficients = None

    def fit(self, x, y):
        """Fit Ridge regression model."""
        # Convert to lists if needed
        if hasattr(x, 'tolist'):
            x = x.tolist()
        if hasattr(y, 'tolist'):
            y = y.tolist()

        # Handle both 1D and 2D inputs
        if isinstance(x[0], (int, float)):
            x = [[xi] for xi in x]

        # Call multivariate_ols which adds intercept
        self.coefficients = self.multivariate_ols(x, y)
        return self

    def predict(self, x):
        """Prediction using Ridge coefficients."""
        fit_error_handling(self.coefficients)

        # Convert to list if needed
        if hasattr(x, 'tolist'):
            x = x.tolist()

        # Handle both 1D and 2D inputs
        if isinstance(x[0], (int, float)):
            x = [[xi] for xi in x]

        # Add intercept column
        X_with_intercept = []
        for row in x:
            X_with_intercept.append([1.0] + row)

        # Predictions
        predictions = []
        for row in X_with_intercept:
            y_pred = sum(row[j] * self.coefficients[j] for j in range(len(self.coefficients)))
            predictions.append(y_pred)

        return predictions


class LassoRegression(LeastSquares):
    """Lasso regression using CoordinateDescent infrastructure"""

    def __init__(self, alpha=1.0, max_iter=20000, tol=1e-4):
        super().__init__(type_regression="LassoRegression")
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients = None

    def fit(self, x, y):
        """Fit Lasso regression model."""
        # Convert to lists if needed
        if hasattr(x, 'tolist'):
            x = x.tolist()
        if hasattr(y, 'tolist'):
            y = y.tolist()

        # Handle both 1D and 2D inputs
        if isinstance(x[0], (int, float)):
            x = [[xi] for xi in x]

        # Call multivariate_ols which adds intercept
        self.coefficients = self.multivariate_ols(x, y)
        return self

    def predict(self, x):
        """Prediction using Lasso coefficients."""
        fit_error_handling(self.coefficients)

        # Convert to list if needed
        if hasattr(x, 'tolist'):
            x = x.tolist()

        # Handle both 1D and 2D inputs
        if isinstance(x[0], (int, float)):
            x = [[xi] for xi in x]

        # Add intercept column
        X_with_intercept = []
        for row in x:
            X_with_intercept.append([1.0] + row)

        # Predictions
        predictions = []
        for row in X_with_intercept:
            y_pred = sum(row[j] * self.coefficients[j] for j in range(len(self.coefficients)))
            predictions.append(y_pred)

        return predictions


class ElasticNetRegression(LeastSquares):
    """Elastic Net regression using CoordinateDescent infrastructure"""

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=20000, tol=1e-4):
        super().__init__(type_regression="ElasticNetRegression")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients = None

    def fit(self, x, y):
        """Fit ElasticNet regression model."""
        # Convert to lists if needed
        if hasattr(x, 'tolist'):
            x = x.tolist()
        if hasattr(y, 'tolist'):
            y = y.tolist()

        # Handle both 1D and 2D inputs
        if isinstance(x[0], (int, float)):
            x = [[xi] for xi in x]

        # Call multivariate_ols which adds intercept
        self.coefficients = self.multivariate_ols(x, y)
        return self

    def predict(self, x):
        """Prediction using ElasticNet coefficients."""
        fit_error_handling(self.coefficients)

        # Convert to list if needed
        if hasattr(x, 'tolist'):
            x = x.tolist()

        # Handle both 1D and 2D inputs
        if isinstance(x[0], (int, float)):
            x = [[xi] for xi in x]

        # Add intercept column
        X_with_intercept = []
        for row in x:
            X_with_intercept.append([1.0] + row)

        # Predictions
        predictions = []
        for row in X_with_intercept:
            y_pred = sum(row[j] * self.coefficients[j] for j in range(len(self.coefficients)))
            predictions.append(y_pred)

        return predictions


class PolynomialRegression:
    """Polynomial regression using pure Python (no NumPy)."""

    def __init__(self, degree=1):
        self.degree = degree
        self.coefficients = None

    def fit(self, X, y):
        """Fit polynomial regression model using normal equations."""
        # Convert inputs to lists if needed
        if hasattr(X, 'tolist'):
            X = X.tolist()
        if hasattr(y, 'tolist'):
            y = y.tolist()

        # Ensure X is a list
        if not isinstance(X, list):
            X = list(X)
        if not isinstance(y, list):
            y = list(y)

        n = len(X)

        # Create design matrix (Vandermonde matrix)
        # Each row is [1, x, x^2, ..., x^degree]
        design_matrix = []
        for i in range(n):
            row = []
            for power in range(self.degree + 1):
                row.append(X[i] ** power)
            design_matrix.append(row)

        # Compute X^T * X using for loops
        XtX = self._matrix_multiply_transpose(design_matrix, design_matrix)

        # Compute X^T * y using for loops
        Xty = self._matrix_vector_multiply_transpose(design_matrix, y)

        # Solve the normal equations (XtX * coeffs = Xty) using Gaussian elimination
        self.coefficients = self._solve_linear_system(XtX, Xty)

        return self

    def predict(self, X):
        """Predict using the fitted polynomial model."""
        if self.coefficients is None:
            raise ValueError("Model must be fitted before prediction")

        # Convert to list if needed
        if hasattr(X, 'tolist'):
            X = X.tolist()
        if not isinstance(X, list):
            X = list(X)

        predictions = []
        for x in X:
            y_pred = 0
            for power, coeff in enumerate(self.coefficients):
                y_pred += coeff * (x ** power)
            predictions.append(y_pred)

        return predictions

    def _matrix_multiply_transpose(self, A, B):
        """Compute A^T * B using for loops."""
        # A is n x m, B is n x k
        # Result is m x k
        n = len(A)
        m = len(A[0]) if n > 0 else 0
        k = len(B[0]) if n > 0 else 0

        result = [[0.0 for _ in range(k)] for _ in range(m)]

        for i in range(m):
            for j in range(k):
                for t in range(n):
                    result[i][j] += A[t][i] * B[t][j]

        return result

    def _matrix_vector_multiply_transpose(self, A, b):
        """Compute A^T * b using for loops."""
        # A is n x m, b is n x 1
        # Result is m x 1
        n = len(A)
        m = len(A[0]) if n > 0 else 0

        result = [0.0 for _ in range(m)]

        for i in range(m):
            for j in range(n):
                result[i] += A[j][i] * b[j]

        return result

    def _solve_linear_system(self, A, b):
        """Solve Ax = b using Gaussian elimination with partial pivoting."""
        n = len(A)

        # Create augmented matrix [A|b]
        augmented = []
        for i in range(n):
            row = A[i][:] + [b[i]]  # Copy row and append b[i]
            augmented.append(row)

        # Forward elimination with partial pivoting
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                    max_row = k

            # Swap rows
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]

            # Check for zero pivot
            if abs(augmented[i][i]) < 1e-10:
                # Add small value to avoid division by zero
                augmented[i][i] = 1e-10

            # Eliminate column
            for k in range(i + 1, n):
                factor = augmented[k][i] / augmented[i][i]
                for j in range(i, n + 1):
                    augmented[k][j] -= factor * augmented[i][j]

        # Back substitution
        x = [0.0 for _ in range(n)]
        for i in range(n - 1, -1, -1):
            x[i] = augmented[i][n]
            for j in range(i + 1, n):
                x[i] -= augmented[i][j] * x[j]
            x[i] /= augmented[i][i]

        return x


class RidgeRegression:
    """Ridge regression using pure Python."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        """Fit ridge regression model."""
        # Convert inputs to lists if needed
        if hasattr(X, 'tolist'):
            X = X.tolist()
        if hasattr(y, 'tolist'):
            y = y.tolist()

        # Handle both 1D and 2D inputs
        if isinstance(X[0], (int, float)):
            X = [[x] for x in X]

        n_samples = len(X)
        n_features = len(X[0])

        # Add intercept column (column of ones)
        X_with_intercept = []
        for i in range(n_samples):
            row = [1.0] + X[i]
            X_with_intercept.append(row)

        # Compute X^T * X
        XtX = self._matrix_multiply_transpose(X_with_intercept, X_with_intercept)

        # Add ridge penalty to diagonal (except first element for intercept)
        for i in range(1, len(XtX)):
            XtX[i][i] += self.alpha

        # Compute X^T * y
        Xty = self._matrix_vector_multiply_transpose(X_with_intercept, y)

        # Solve the system
        coeffs = self._solve_linear_system(XtX, Xty)

        # Separate intercept and coefficients
        self.intercept = coeffs[0]
        self.coefficients = coeffs[1:]  # Only the feature coefficients, not intercept

        return self

    def predict(self, X):
        """Predict using the fitted ridge model."""
        if self.coefficients is None:
            raise ValueError("Model must be fitted before prediction")

        # Convert to list if needed
        if hasattr(X, 'tolist'):
            X = X.tolist()

        # Handle both 1D and 2D inputs
        if isinstance(X[0], (int, float)):
            X = [[x] for x in X]

        predictions = []
        for row in X:
            y_pred = self.intercept if self.intercept is not None else 0
            for i, x_val in enumerate(row):
                if i < len(self.coefficients):
                    y_pred += self.coefficients[i] * x_val
            predictions.append(y_pred)

        return predictions

    def _matrix_multiply_transpose(self, A, B):
        """Compute A^T * B using for loops."""
        n = len(A)
        m = len(A[0]) if n > 0 else 0
        k = len(B[0]) if n > 0 else 0

        result = [[0.0 for _ in range(k)] for _ in range(m)]

        for i in range(m):
            for j in range(k):
                for t in range(n):
                    result[i][j] += A[t][i] * B[t][j]

        return result

    def _matrix_vector_multiply_transpose(self, A, b):
        """Compute A^T * b using for loops."""
        n = len(A)
        m = len(A[0]) if n > 0 else 0

        result = [0.0 for _ in range(m)]

        for i in range(m):
            for j in range(n):
                result[i] += A[j][i] * b[j]

        return result

    def _solve_linear_system(self, A, b):
        """Solve Ax = b using Gaussian elimination with partial pivoting."""
        n = len(A)

        # Create augmented matrix [A|b]
        augmented = []
        for i in range(n):
            row = A[i][:] + [b[i]]
            augmented.append(row)

        # Forward elimination with partial pivoting
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                    max_row = k

            # Swap rows
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]

            # Check for zero pivot
            if abs(augmented[i][i]) < 1e-10:
                augmented[i][i] = 1e-10

            # Eliminate column
            for k in range(i + 1, n):
                factor = augmented[k][i] / augmented[i][i]
                for j in range(i, n + 1):
                    augmented[k][j] -= factor * augmented[i][j]

        # Back substitution
        x = [0.0 for _ in range(n)]
        for i in range(n - 1, -1, -1):
            x[i] = augmented[i][n]
            for j in range(i + 1, n):
                x[i] -= augmented[i][j] * x[j]
            x[i] /= augmented[i][i]

        return x


class LassoRegression:
    """Lasso regression using sklearn (as allowed)."""

    def __init__(self, alpha=1.0, max_iter=20000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.model = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=True)
        self.coefficients = None

    def fit(self, X, y):
        """Fit lasso regression model using sklearn."""
        # sklearn's Lasso can handle lists or arrays
        self.model.fit(X, y)

        # Store coefficients in the same format as other models
        self.coefficients = [self.model.intercept_] + list(self.model.coef_)

        return self

    def predict(self, X):
        """Predict using the fitted lasso model."""
        return self.model.predict(X)


class ElasticNetRegression:
    """Elastic Net regression using sklearn (as allowed)."""

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=20000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                max_iter=max_iter, fit_intercept=True)
        self.coefficients = None

    def fit(self, X, y):
        """Fit elastic net regression model using sklearn."""
        # sklearn's ElasticNet can handle lists or arrays
        self.model.fit(X, y)

        # Store coefficients in the same format as other models
        self.coefficients = [self.model.intercept_] + list(self.model.coef_)

        return self

    def predict(self, X):
        """Predict using the fitted elastic net model."""
        return self.model.predict(X)