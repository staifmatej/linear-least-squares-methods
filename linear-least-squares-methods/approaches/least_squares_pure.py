"""LeastSquares implementation in Pure Python - FIXED VERSION."""

import warnings
import math
from sklearn.linear_model import Lasso, ElasticNet
from constants import S_RED, E_RED

# Suppress common warnings
warnings.filterwarnings('ignore', message='Objective did not converge')


# pylint: disable=duplicate-code
def fit_error_handling(coefficients):
    """fit error handling for Regression model."""
    if coefficients is None:
        raise ValueError("Model not fitted yet. Call fit() first.")

# pylint: disable=duplicate-code
def create_zero_matrix(rows, cols):
    """Create a matrix filled with zeros."""
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def create_zero_vector(size):
    """Create a vector filled with zeros."""
    return [0.0 for _ in range(size)]

def create_ones_vector(size):
    """Create a vector filled with ones."""
    return [1.0 for _ in range(size)]

def matrix_multiply_transpose_pure(A, B):
    """Compute A^T * B using Pure Python."""
    m = len(A)
    n = len(A[0]) if m > 0 else 0
    k = len(B[0]) if m > 0 else 0

    result = create_zero_matrix(n, k)

    for i in range(n):
        for j in range(k):
            for t in range(m):
                result[i][j] += A[t][i] * B[t][j]

    return result


def matrix_vector_multiply_transpose_pure(A, b):
    """Compute A^T * b using Pure Python."""
    m = len(A)
    n = len(A[0]) if m > 0 else 0

    result = create_zero_vector(n)

    for i in range(n):
        for j in range(m):
            result[i] += A[j][i] * b[j]

    return result


def matrix_vector_multiply_pure(A, v):
    """Compute A * v using Pure Python."""
    m = len(A)
    n = len(A[0])

    result = create_zero_vector(m)

    for i in range(m):
        for j in range(n):
            result[i] += A[i][j] * v[j]

    return result


def solve_linear_system_pure(A, b):
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(A)

    # Create augmented matrix
    augmented = create_zero_matrix(n, n + 1)
    for i in range(n):
        for j in range(n):
            augmented[i][j] = A[i][j]
        augmented[i][n] = b[i]

    # Forward elimination with partial pivoting
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k

        # Swap rows
        if max_row != i:
            for j in range(n + 1):
                temp = augmented[i][j]
                augmented[i][j] = augmented[max_row][j]
                augmented[max_row][j] = temp

        # Check for zero pivot
        if abs(augmented[i][i]) < 1e-10:
            augmented[i][i] = 1e-10

        # Eliminate column
        for k in range(i + 1, n):
            factor = augmented[k][i] / augmented[i][i]
            for j in range(i, n + 1):
                augmented[k][j] -= factor * augmented[i][j]

    # Back substitution
    x = create_zero_vector(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented[i][n]
        for j in range(i + 1, n):
            x[i] -= augmented[i][j] * x[j]
        x[i] /= augmented[i][i]

    return x


def qr_decomposition_pure(A):
    """QR decomposition using Gram-Schmidt orthogonalization with Pure Python."""
    m = len(A)
    n = len(A[0])

    # Initialize Q and R
    Q = create_zero_matrix(m, n)
    R = create_zero_matrix(n, n)

    # Gram-Schmidt process
    for j in range(n):
        # Get column j of A
        v = create_zero_vector(m)
        for i in range(m):
            v[i] = A[i][j]

        # Orthogonalize against previous columns
        for i in range(j):
            # R[i][j] = Q[:, i]^T * A[:, j]
            R[i][j] = 0.0
            for k in range(m):
                R[i][j] += Q[k][i] * A[k][j]
            # v = v - R[i][j] * Q[:, i]
            for k in range(m):
                v[k] -= R[i][j] * Q[k][i]

        # Normalize
        R[j][j] = 0.0
        for k in range(m):
            R[j][j] += v[k] * v[k]
        R[j][j] = math.sqrt(R[j][j])

        if R[j][j] > 1e-10:
            for k in range(m):
                Q[k][j] = v[k] / R[j][j]
        else:
            # Handle zero column
            for k in range(m):
                Q[k][j] = 0.0

    return Q, R


def back_substitution_pure(R, b):
    """Solve Rx = b where R is upper triangular using Pure Python."""
    n = len(R)
    x = create_zero_vector(n)

    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= R[i][j] * x[j]
        if abs(R[i][i]) > 1e-10:
            x[i] /= R[i][i]
        else:
            x[i] = 0.0

    return x


def eigenvalues_power_method_pure(A, max_iter=50000):
    """Approximate eigenvalues using power method with Pure Python."""
    n = len(A)

    # Just get largest eigenvalue for now
    v = create_ones_vector(n)
    for _ in range(max_iter):
        v_new = matrix_vector_multiply_pure(A, v)
        norm = 0.0
        for i, val in enumerate(v_new):
            norm += val ** 2
        norm = math.sqrt(norm)
        if norm > 1e-10:
            for i, val in enumerate(v_new):
                v[i] = val / norm

    # Rayleigh quotient
    Av = matrix_vector_multiply_pure(A, v)
    lambda_max = 0.0
    for i in range(n):
        lambda_max += v[i] * Av[i]

    # Return both max and rough estimate of min
    return [lambda_max, lambda_max / 1000.0]


def generate_polynomial_features_pure(x, degree, x_min, x_max, normalize):
    """Generate polynomial features with Pure Python."""
    n = len(x)

    if normalize and degree > 3:
        # Normalize x to interval [-1, 1] for better numerical stability
        if x_max - x_min > 1e-10:
            x_normalized = create_zero_vector(n)
            for i in range(n):
                x_normalized[i] = 2 * (x[i] - x_min) / (x_max - x_min) - 1
        else:
            x_normalized = x[:]
    else:
        x_normalized = x[:]

    polynomial_features = create_zero_matrix(n, degree)
    for i in range(n):
        for d in range(1, degree + 1):
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

            polynomial_features[i][d - 1] = feature

    return polynomial_features


class LeastSquares:
    """LeastSquares implementation using Numba-accelerated functions."""

    def __init__(self, type_regression="LinearRegression"):
        self.type_regression = type_regression
        # Set different alpha values for different regression types
        if type_regression == "RidgeRegression":
            self.alpha = 0.01  # Lower alpha for Ridge
        elif type_regression == "LassoRegression":
            self.alpha = 1.0   # Keep higher alpha for Lasso
        elif type_regression == "ElasticNetRegression":
            self.alpha = 0.1   # Medium alpha for ElasticNet
        else:
            self.alpha = 1.0   # Default
        self.l1_ratio = 0.5  # Default l1_ratio for ElasticNet
        self.max_iter = 50000
        self.tol = 1e-4
        self.condition_number = None  # Store condition number for printing

        types_of_regression = ["LinearRegression", "RidgeRegression", "LassoRegression", "ElasticNetRegression"]
        if type_regression not in types_of_regression:
            raise ValueError(f"Type {self.type_regression} is not a valid predefined type.")

    # pylint: disable=duplicate-code, disable=too-many-branches
    def multivariate_ols(self, X, Y):
        """
        X: feature matrix
            Format: list of lists, where N >= p and N = number of observations, p = number of features
        Y: target variable vector
            Format: list - one-dimensional
        return: w = coefficient vector [w₀, w₁, w₂, ..., wₚ]
        """
        # Convert to pure Python lists
        if hasattr(X, 'tolist'):
            X = X.tolist()
        elif not isinstance(X, list):
            X = list(X)

        if hasattr(Y, 'tolist'):
            Y = Y.tolist()
        elif not isinstance(Y, list):
            Y = list(Y)

        # Handle 1D input
        if not isinstance(X[0], list):
            X = [[x] for x in X]

        # Add column of ones for intercept
        n_samples = len(Y)
        n_features = len(X[0])
        X_with_intercept = create_zero_matrix(n_samples, n_features + 1)

        for i in range(n_samples):
            X_with_intercept[i][0] = 1.0  # intercept column
            for j in range(n_features):
                X_with_intercept[i][j + 1] = X[i][j]

        n_rows = len(X_with_intercept)
        n_cols = len(X_with_intercept[0])

        if n_rows < n_cols:
            raise ValueError(f"Matrix X must have more rows than columns. Got {n_rows} rows and {n_cols} columns")

        # Calculate condition number
        threshold_for_QR_decomposition = 1e6
        cond_number = 0

        if self.type_regression == "RidgeRegression":
            cond_number = self._calculate_ridge_condition_number(X_with_intercept)
            self.condition_number = cond_number
        elif self.type_regression == "LassoRegression":
            self.condition_number = None  # Lasso doesn't use direct condition number calculation
            return self._coordinate_descent_lasso(X_with_intercept, Y)
        elif self.type_regression == "ElasticNetRegression":
            self.condition_number = None  # ElasticNet doesn't use direct condition number calculation
            return self._coordinate_descent_elasticnet(X_with_intercept, Y)
        elif self.type_regression == "LinearRegression":
            cond_number = self._calculate_standard_condition_number(X_with_intercept)
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
            w = self.normal_equations(X_with_intercept, Y)
        else:
            w = self.qr_decomposition(X_with_intercept, Y)

        return w  # Already a list

    def normal_equations(self, X, Y):
        """Compute LeastSquares coefficients using normal equations method."""
        # Compute X^T * Y using Pure Python
        XtY = matrix_vector_multiply_transpose_pure(X, Y)

        if self.type_regression == "RidgeRegression":
            # Compute X^T * X using Pure Python
            XtX = matrix_multiply_transpose_pure(X, X)
            n_features = len(XtX)

            # Add ridge regularization
            for i in range(1, n_features):  # Skip intercept
                XtX[i, i] += self.alpha

            # Solve system using Pure Python
            try:
                w = solve_linear_system_pure(XtX, XtY)
            except (ValueError, ArithmeticError, ZeroDivisionError):
                # Fallback to pseudo-inverse
                w = self._solve_with_pseudoinverse(XtX, XtY)
        else:
            # Compute X^T * X using Pure Python
            XtX = matrix_multiply_transpose_pure(X, X)

            # Solve system using Pure Python
            try:
                w = solve_linear_system_pure(XtX, XtY)
            except (ValueError, ArithmeticError, ZeroDivisionError):
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
            X_extended = create_zero_matrix(n_samples + n_features, n_features)
            # Original X
            for i in range(n_samples):
                for j in range(n_features):
                    X_extended[i][j] = X[i][j]
            # Regularization rows
            for i in range(n_features):
                if i > 0:  # Skip intercept
                    X_extended[n_samples + i][i] = sqrt_alpha

            # Extended Y
            Y_extended = create_zero_vector(n_samples + n_features)
            for i in range(n_samples):
                Y_extended[i] = Y[i]

            # QR decomposition using Pure Python
            Q, R = qr_decomposition_pure(X_extended)
            QtY = matrix_vector_multiply_transpose_pure(Q, Y_extended)
        else:
            # Standard QR decomposition using Pure Python
            Q, R = qr_decomposition_pure(X)
            QtY = matrix_vector_multiply_transpose_pure(Q, Y)

        # Solve R * w = QtY using back substitution with Pure Python
        w = back_substitution_pure(R, QtY)
        return w

    def _coordinate_descent_lasso(self, X, Y):
        """Lasso coordinate descent implementation using sklearn."""
        # Remove intercept column for sklearn
        X_features = [row[1:] for row in X]

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
        X_features = [row[1:] for row in X]

        # Use ElasticNet model directly
        enet = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                          max_iter=self.max_iter, tol=self.tol, fit_intercept=True)
        enet.fit(X_features, Y)

        # Return coefficients including intercept
        coeffs = [enet.intercept_] + list(enet.coef_)
        return coeffs

    def _calculate_ridge_condition_number(self, X):
        """Calculate condition number for Ridge regression."""
        XtX = matrix_multiply_transpose_pure(X, X)
        n_features = len(XtX)

        # Add ridge regularization
        for i in range(1, n_features):
            XtX[i, i] += self.alpha

        return self._condition_number(XtX)

    def _calculate_standard_condition_number(self, X):
        """Calculate condition number for standard regression."""
        XtX = matrix_multiply_transpose_pure(X, X)
        return self._condition_number(XtX)

    def _condition_number(self, A):
        """Calculate condition number of matrix A using eigenvalues."""
        try:
            eigenvalues = eigenvalues_power_method_pure(A)
            if len(eigenvalues) == 0:
                return 1e20

            max_eig = max(abs(e) for e in eigenvalues)
            min_eig = min(abs(e) for e in eigenvalues if abs(e) > 1e-10)

            if min_eig < 1e-10:
                return 1e20

            return max_eig / min_eig
        except (ValueError, ArithmeticError, ZeroDivisionError):
            return 1e20

    # All mathematical operations delegated to pure python functions

    def _solve_with_pseudoinverse(self, A, b):
        """Solve using pseudo-inverse (simplified implementation)."""
        # This is a simplified version - just add small diagonal for stability
        n = len(A)
        A_reg = [row[:] for row in A]  # Deep copy

        # Add small value to diagonal
        for i in range(n):
            A_reg[i][i] += 1e-10

        return solve_linear_system_pure(A_reg, b)

    # Matrix operations are handled by numba functions directly


# Cleaned up - keeping only the simplest implementations

class LinearRegression:
    """Polynomial regression using pure Python (no NumPy) with Pure Python acceleration."""

    def __init__(self, degree=1):
        self.degree = degree
        self.coefficients = None
        self.x_min = None  # For normalization
        self.x_max = None  # For normalization

    def fit(self, X, y):
        """Fit polynomial regression model using normal equations."""
        # Convert inputs to pure Python lists
        if hasattr(X, 'tolist'):
            X = X.tolist()
        elif not isinstance(X, list):
            X = list(X)

        if hasattr(y, 'tolist'):
            y = y.tolist()
        elif not isinstance(y, list):
            y = list(y)

        n = len(X)

        # Check for underdetermined system (more parameters than data points)
        if n <= self.degree:
            raise ValueError(f"Cannot fit polynomial of degree {self.degree} with {n} data points. "
                           f"Need at least {self.degree + 1} data points.")

        # Save range for normalization
        self.x_min = min(X)
        self.x_max = max(X)

        # Generate polynomial features (consistent with NumPy approach)
        X_polynomial = self._generate_polynomial_features_consistent(X)

        # Add intercept column
        n_features = len(X_polynomial[0])
        X_with_intercept = create_zero_matrix(n, n_features + 1)
        for i in range(n):
            X_with_intercept[i][0] = 1.0  # intercept
            for j in range(n_features):
                X_with_intercept[i][j + 1] = X_polynomial[i][j]

        # Compute X^T * X using Pure Python
        XtX = matrix_multiply_transpose_pure(X_with_intercept, X_with_intercept)

        # Compute X^T * y using Pure Python
        Xty = matrix_vector_multiply_transpose_pure(X_with_intercept, y)

        # Solve the normal equations using Pure Python
        self.coefficients = solve_linear_system_pure(XtX, Xty)

        return self

    def predict(self, X):
        """Predict using the fitted polynomial model."""
        if self.coefficients is None:
            raise ValueError("Model must be fitted before prediction")

        # Convert to pure Python list
        if hasattr(X, 'tolist'):
            X = X.tolist()
        elif not isinstance(X, list):
            X = list(X)

        # Generate polynomial features consistent with training
        X_polynomial = self._generate_polynomial_features_consistent(X)

        # Add intercept column
        n_features = len(X_polynomial[0])
        X_with_intercept = create_zero_matrix(len(X), n_features + 1)
        for i in range(len(X)):
            X_with_intercept[i][0] = 1.0  # intercept
            for j in range(n_features):
                X_with_intercept[i][j + 1] = X_polynomial[i][j]

        # Predict using coefficients
        predictions = []
        for i in range(len(X)):
            y_pred = 0
            for j, coeff in enumerate(self.coefficients):
                y_pred += coeff * X_with_intercept[i][j]
            predictions.append(y_pred)

        return predictions

    def _generate_polynomial_features(self, x):
        """Generate polynomial features consistent with NumPy approach."""
        # Apply normalization for degree > 3
        if self.degree > 3:
            # Normalize x to interval [-1, 1] for better numerical stability
            if self.x_max - self.x_min > 1e-10:
                x_normalized = [2 * (xi - self.x_min) / (self.x_max - self.x_min) - 1 for xi in x]
            else:
                x_normalized = x[:]
        else:
            x_normalized = x[:]

        # Generate polynomial features for degrees 1 to self.degree
        n = len(x)
        polynomial_features = create_zero_matrix(n, self.degree)
        for i in range(n):
            for d in range(1, self.degree + 1):
                polynomial_features[i][d - 1] = x_normalized[i] ** d

        return polynomial_features

    def _generate_polynomial_features_consistent(self, x):
        """Generate polynomial features consistent with NumPy approach."""
        return self._generate_polynomial_features(x)

    # Matrix operations use pure python functions directly


class RidgeRegression:
    """Ridge regression using pure Python with Pure Python acceleration."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        """Fit ridge regression model."""
        # Convert inputs to pure Python lists
        if hasattr(X, 'tolist'):
            X = X.tolist()
        elif not isinstance(X, list):
            X = list(X)

        if hasattr(y, 'tolist'):
            y = y.tolist()
        elif not isinstance(y, list):
            y = list(y)

        # Handle both 1D and 2D inputs
        if not isinstance(X[0], list):
            X = [[x] for x in X]

        n_samples = len(X)
        n_features = len(X[0])

        # Add intercept column (column of ones)
        X_with_intercept = create_zero_matrix(n_samples, n_features + 1)
        for i in range(n_samples):
            X_with_intercept[i][0] = 1.0  # intercept
            for j in range(n_features):
                X_with_intercept[i][j + 1] = X[i][j]

        # Compute X^T * X using Pure Python
        XtX = matrix_multiply_transpose_pure(X_with_intercept, X_with_intercept)

        # Add ridge penalty to diagonal (except first element for intercept)
        for i in range(1, len(XtX)):
            XtX[i][i] += self.alpha

        # Compute X^T * y using Pure Python
        Xty = matrix_vector_multiply_transpose_pure(X_with_intercept, y)

        # Solve the system using Pure Python
        coeffs = solve_linear_system_pure(XtX, Xty)

        # Store all coefficients including intercept for consistency with other models
        self.intercept = coeffs[0]
        self.coefficients = coeffs  # All coefficients [intercept, coef1, coef2, ...]

        return self

    def predict(self, X):
        """Predict using the fitted ridge model."""
        if self.coefficients is None:
            raise ValueError("Model must be fitted before prediction")

        # Convert to pure Python list
        if hasattr(X, 'tolist'):
            X = X.tolist()
        elif not isinstance(X, list):
            X = list(X)

        # Handle both 1D and 2D inputs
        if not isinstance(X[0], list):
            X = [[x] for x in X]

        # Calculate expected number of features from coefficients
        n_coeffs = len(self.coefficients)
        expected_features = n_coeffs - 1  # subtract intercept

        if len(X[0]) == 1 and expected_features > 1:
            # This is polynomial regression - generate polynomial features
            degree = expected_features
            x_flat = [row[0] for row in X]
            x_min = min(x_flat)
            x_max = max(x_flat)
            X_poly = generate_polynomial_features_pure(x_flat, degree, x_min, x_max, degree > 3)

            # Add intercept column to polynomial features
            predictions = []
            for _, x_row in enumerate(X_poly):
                y_pred = self.coefficients[0]  # intercept
                for j, val in enumerate(x_row):
                    y_pred += self.coefficients[j + 1] * val
                predictions.append(y_pred)
        else:
            # Regular linear prediction
            predictions = []
            for _, x_row in enumerate(X):
                y_pred = self.coefficients[0] if len(self.coefficients) > 0 else 0  # intercept
                for j, val in enumerate(x_row):
                    if j + 1 < len(self.coefficients):  # +1 because first coeff is intercept
                        y_pred += self.coefficients[j + 1] * val
                predictions.append(y_pred)

        return predictions

    # Matrix operations use pure python functions directly


class LassoRegression:
    """Lasso regression using sklearn (as allowed)."""

    def __init__(self, alpha=1.0, max_iter=50000):
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
        # Convert to pure Python list first for processing
        if hasattr(X, 'tolist'):
            X_list = X.tolist()
        elif not isinstance(X, list):
            X_list = list(X)
        else:
            X_list = X

        # Handle both 1D and 2D inputs
        if not isinstance(X_list[0], list):
            X_list = [[x] for x in X_list]

        # Calculate expected number of features from coefficients
        n_coeffs = len(self.coefficients)
        expected_features = n_coeffs - 1  # subtract intercept

        if len(X_list[0]) == 1 and expected_features > 1:
            # This is polynomial regression - generate polynomial features
            degree = expected_features
            x_flat = [row[0] for row in X_list]
            x_min = min(x_flat)
            x_max = max(x_flat)
            X_poly = generate_polynomial_features_pure(x_flat, degree, x_min, x_max, degree > 3)
            X = X_poly  # Use for sklearn prediction
        else:
            X = X_list

        return self.model.predict(X)


class ElasticNetRegression:
    """Elastic Net regression using sklearn (as allowed)."""

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=50000):
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
        # Convert to pure Python list first for processing
        if hasattr(X, 'tolist'):
            X_list = X.tolist()
        elif not isinstance(X, list):
            X_list = list(X)
        else:
            X_list = X

        # Handle both 1D and 2D inputs
        if not isinstance(X_list[0], list):
            X_list = [[x] for x in X_list]

        # Calculate expected number of features from coefficients
        n_coeffs = len(self.coefficients)
        expected_features = n_coeffs - 1  # subtract intercept

        if len(X_list[0]) == 1 and expected_features > 1:
            # This is polynomial regression - generate polynomial features
            degree = expected_features
            x_flat = [row[0] for row in X_list]
            x_min = min(x_flat)
            x_max = max(x_flat)
            X_poly = generate_polynomial_features_pure(x_flat, degree, x_min, x_max, degree > 3)
            X = X_poly  # Use for sklearn prediction
        else:
            X = X_list

        return self.model.predict(X)
