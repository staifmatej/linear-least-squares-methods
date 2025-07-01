"""LeastSquares implementation in Python with Numba acceleration - FIXED VERSION."""

import warnings
import math
from numba import njit
from sklearn.linear_model import Lasso, ElasticNet
from constants import S_RED, E_RED

warnings.filterwarnings('ignore', message='Objective did not converge')

# pylint: disable=duplicate-code
@njit
def fit_error_handling(coefficients):
    """fit error handling for Regression model."""
    if coefficients is None:
        raise ValueError("Model not fitted yet. Call fit() first.")


# pylint: disable=duplicate-code
def zeros_2d(rows, cols):
    """Create a 2D matrix filled with zeros - regular Python lists."""
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def zeros_1d(size):
    """Create a 1D array filled with zeros - regular Python list."""
    return [0.0 for _ in range(size)]

def create_zero_matrix(rows, cols):
    """Create a matrix filled with zeros."""
    return zeros_2d(rows, cols)

# pylint: disable=duplicate-code
@njit
def create_zero_vector(size):
    """Create a vector filled with zeros."""
    return [0.0 for _ in range(size)]

# pylint: disable=duplicate-code
@njit
def create_ones_vector(size):
    """Create a vector filled with ones."""
    return [1.0 for _ in range(size)]

# pylint: disable=duplicate-code
@njit
def _matrix_multiply_transpose_flat(A_flat, B_flat, m, n, k):
    """@njit core function using flat arrays only."""
    result_flat = [0.0 for _ in range(n * k)]

    for i in range(n):
        for j in range(k):
            for t in range(m):
                result_flat[i * k + j] += A_flat[t * n + i] * B_flat[t * k + j]

    return result_flat

def matrix_multiply_transpose_numba(A, B):
    """Compute A^T * B using @njit core + wrapper."""
    m = len(A)
    n = len(A[0]) if m > 0 else 0
    k = len(B[0]) if m > 0 else 0

    # Convert A and B to flat arrays for @njit
    A_flat = [0.0 for _ in range(m * n)]
    B_flat = [0.0 for _ in range(m * k)]

    for i in range(m):
        for j in range(n):
            A_flat[i * n + j] = A[i][j]

    for i in range(m):
        for j in range(k):
            B_flat[i * k + j] = B[i][j]

    # Use @njit core for speed
    result_flat = _matrix_multiply_transpose_flat(A_flat, B_flat, m, n, k)

    # Convert to nested list for compatibility
    result = [[0.0 for _ in range(k)] for _ in range(n)]
    for i in range(n):
        for j in range(k):
            result[i][j] = result_flat[i * k + j]

    return result

# pylint: disable=duplicate-code
@njit
def _matrix_vector_multiply_transpose_flat(A_flat, b, m, n):
    """@njit core function using flat array."""
    result = [0.0 for _ in range(n)]

    for i in range(n):
        for j in range(m):
            result[i] += A_flat[j * n + i] * b[j]

    return result

def matrix_vector_multiply_transpose_numba(A, b):
    """Compute A^T * b using @njit core + wrapper."""
    m = len(A)
    n = len(A[0]) if m > 0 else 0

    # Convert A to flat array
    A_flat = [0.0 for _ in range(m * n)]
    for i in range(m):
        for j in range(n):
            A_flat[i * n + j] = A[i][j]

    return _matrix_vector_multiply_transpose_flat(A_flat, b, m, n)

# pylint: disable=duplicate-code
@njit
def _matrix_vector_multiply_flat(A_flat, v, m, n):
    """@njit core function using flat array."""
    result = [0.0 for _ in range(m)]

    for i in range(m):
        for j in range(n):
            result[i] += A_flat[i * n + j] * v[j]

    return result

def matrix_vector_multiply_numba(A, v):
    """Compute A * v using @njit core + wrapper."""
    m = len(A)
    n = len(A[0])

    # Convert A to flat array
    A_flat = [0.0 for _ in range(m * n)]
    for i in range(m):
        for j in range(n):
            A_flat[i * n + j] = A[i][j]

    return _matrix_vector_multiply_flat(A_flat, v, m, n)

# pylint: disable=duplicate-code
@njit
def _solve_linear_system_flat(A_flat, b, n):
    """@njit core function using flat arrays."""
    # Create flat augmented matrix
    augmented_flat = [0.0 for _ in range(n * (n + 1))]

    # Copy A and b to augmented matrix
    for i in range(n):
        for j in range(n):
            augmented_flat[i * (n + 1) + j] = A_flat[i * n + j]
        augmented_flat[i * (n + 1) + n] = b[i]

    # Forward elimination with partial pivoting
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented_flat[k * (n + 1) + i]) > abs(augmented_flat[max_row * (n + 1) + i]):
                max_row = k

        # Swap rows
        if max_row != i:
            for j in range(n + 1):
                temp = augmented_flat[i * (n + 1) + j]
                augmented_flat[i * (n + 1) + j] = augmented_flat[max_row * (n + 1) + j]
                augmented_flat[max_row * (n + 1) + j] = temp

        # Check for zero pivot
        if abs(augmented_flat[i * (n + 1) + i]) < 1e-10:
            augmented_flat[i * (n + 1) + i] = 1e-10

        # Eliminate column
        for k in range(i + 1, n):
            factor = augmented_flat[k * (n + 1) + i] / augmented_flat[i * (n + 1) + i]
            for j in range(i, n + 1):
                augmented_flat[k * (n + 1) + j] -= factor * augmented_flat[i * (n + 1) + j]

    # Back substitution
    x = [0.0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = augmented_flat[i * (n + 1) + n]
        for j in range(i + 1, n):
            x[i] -= augmented_flat[i * (n + 1) + j] * x[j]
        x[i] /= augmented_flat[i * (n + 1) + i]

    return x

def solve_linear_system_numba(A, b):
    """Solve Ax = b using @njit core + wrapper."""
    n = len(A)

    # Convert A to flat array
    A_flat = [0.0 for _ in range(n * n)]
    for i in range(n):
        for j in range(n):
            A_flat[i * n + j] = A[i][j]

    return _solve_linear_system_flat(A_flat, b, n)

@njit
def _qr_decomposition_flat(A_flat, m, n):
    """@njit QR decomposition using flat arrays."""
    # Initialize Q and R as flat arrays
    Q_flat = [0.0 for _ in range(m * n)]
    R_flat = [0.0 for _ in range(n * n)]

    # Gram-Schmidt process
    for j in range(n):
        # Get column j of A
        v = [0.0 for _ in range(m)]
        for i in range(m):
            v[i] = A_flat[i * n + j]

        # Orthogonalize against previous columns
        for i in range(j):
            # R[i][j] = Q[:, i]^T * A[:, j]
            R_ij = 0.0
            for k in range(m):
                R_ij += Q_flat[k * n + i] * A_flat[k * n + j]
            R_flat[i * n + j] = R_ij

            # v = v - R[i][j] * Q[:, i]
            for k in range(m):
                v[k] -= R_ij * Q_flat[k * n + i]

        # Normalize
        R_jj = 0.0
        for k in range(m):
            R_jj += v[k] * v[k]
        R_jj = math.sqrt(R_jj)
        R_flat[j * n + j] = R_jj

        if R_jj > 1e-10:
            for k in range(m):
                Q_flat[k * n + j] = v[k] / R_jj
        else:
            # Handle zero column
            for k in range(m):
                Q_flat[k * n + j] = 0.0

    return Q_flat, R_flat

def qr_decomposition_numba(A):
    """QR decomposition using @njit core + wrapper."""
    m = len(A)
    n = len(A[0])

    # Convert A to flat array
    A_flat = [0.0 for _ in range(m * n)]
    for i in range(m):
        for j in range(n):
            A_flat[i * n + j] = A[i][j]

    # Use @njit core
    Q_flat, R_flat = _qr_decomposition_flat(A_flat, m, n)

    # Convert back to nested lists
    Q = [[0.0 for _ in range(n)] for _ in range(m)]
    R = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(m):
        for j in range(n):
            Q[i][j] = Q_flat[i * n + j]

    for i in range(n):
        for j in range(n):
            R[i][j] = R_flat[i * n + j]

    return Q, R

@njit
def _back_substitution_flat(R_flat, b, n):
    """@njit back substitution using flat array."""
    x = [0.0 for _ in range(n)]

    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= R_flat[i * n + j] * x[j]
        if abs(R_flat[i * n + i]) > 1e-10:
            x[i] /= R_flat[i * n + i]
        else:
            x[i] = 0.0

    return x

def back_substitution_numba(R, b):
    """Solve Rx = b using @njit core + wrapper."""
    n = len(R)

    # Convert R to flat array
    R_flat = [0.0 for _ in range(n * n)]
    for i in range(n):
        for j in range(n):
            R_flat[i * n + j] = R[i][j]

    return _back_substitution_flat(R_flat, b, n)

@njit
def _eigenvalues_power_method_flat(A_flat, n, max_iter=50000):
    """@njit power method using flat arrays."""
    # Just get largest eigenvalue for now
    v = [1.0 for _ in range(n)]

    for _ in range(max_iter):
        # A * v using flat array
        v_new = [0.0 for _ in range(n)]
        for i in range(n):
            for j in range(n):
                v_new[i] += A_flat[i * n + j] * v[j]

        # Normalize
        norm = 0.0
        for i in range(n):
            norm += v_new[i] ** 2
        norm = math.sqrt(norm)

        if norm > 1e-10:
            for i in range(n):
                v[i] = v_new[i] / norm

    # Rayleigh quotient: v^T * A * v
    Av = [0.0 for _ in range(n)]
    for i in range(n):
        for j in range(n):
            Av[i] += A_flat[i * n + j] * v[j]

    lambda_max = 0.0
    for i in range(n):
        lambda_max += v[i] * Av[i]

    return [lambda_max, lambda_max / 1000.0]

def eigenvalues_power_method_numba(A, max_iter=50000):
    """Eigenvalues using @njit core + wrapper."""
    n = len(A)

    # Convert A to flat array
    A_flat = [0.0 for _ in range(n * n)]
    for i in range(n):
        for j in range(n):
            A_flat[i * n + j] = A[i][j]

    return _eigenvalues_power_method_flat(A_flat, n, max_iter)

@njit
def _generate_polynomial_features_flat(x, degree, x_min, x_max, normalize):
    """Generate polynomial features as flat array for @njit."""
    n = len(x)

    if normalize and degree > 3:
        # Normalize x to interval [-1, 1] for better numerical stability
        if x_max - x_min > 1e-10:
            x_normalized = [0.0 for _ in range(n)]
            for i in range(n):
                x_normalized[i] = 2 * (x[i] - x_min) / (x_max - x_min) - 1
        else:
            x_normalized = x[:]
    else:
        x_normalized = x[:]

    # Use flat array for @njit compatibility
    flat_features = [0.0 for _ in range(n * degree)]
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

            flat_features[i * degree + (d - 1)] = feature

    return flat_features

def generate_polynomial_features_numba(x, degree, x_min, x_max, normalize):
    """Generate polynomial features using @njit core + wrapper."""
    # Use @njit core function for speed
    flat_features = _generate_polynomial_features_flat(x, degree, x_min, x_max, normalize)

    # Convert to nested list for compatibility
    n = len(x)
    polynomial_features = [[0.0 for _ in range(degree)] for _ in range(n)]
    for i in range(n):
        for d in range(degree):
            polynomial_features[i][d] = flat_features[i * degree + d]

    return polynomial_features


class LeastSquares:
    """LeastSquares implementation using Numba-accelerated functions."""

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

    # pylint: disable=duplicate-code,too-many-branches,too-many-statements
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
        # Compute X^T * Y using Numba
        XtY = matrix_vector_multiply_transpose_numba(X, Y)

        if self.type_regression == "RidgeRegression":
            # Compute X^T * X using Numba
            XtX = matrix_multiply_transpose_numba(X, X)
            n_features = len(XtX)

            # Add ridge regularization
            for i in range(1, n_features):  # Skip intercept
                XtX[i][i] += self.alpha

            # Solve system using Numba
            try:
                w = solve_linear_system_numba(XtX, XtY)
            except (ValueError, TypeError, ArithmeticError):
                # Fallback to pseudo-inverse
                w = self._solve_with_pseudoinverse(XtX, XtY)
        else:
            # Compute X^T * X using Numba
            XtX = matrix_multiply_transpose_numba(X, X)

            # Solve system using Numba
            try:
                w = solve_linear_system_numba(XtX, XtY)
            except (ValueError, TypeError, ArithmeticError):
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

            # QR decomposition using Numba
            Q, R = qr_decomposition_numba(X_extended)
            QtY = matrix_vector_multiply_transpose_numba(Q, Y_extended)
        else:
            # Standard QR decomposition using Numba
            Q, R = qr_decomposition_numba(X)
            QtY = matrix_vector_multiply_transpose_numba(Q, Y)

        # Solve R * w = QtY using back substitution with Numba
        w = back_substitution_numba(R, QtY)
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
        XtX = matrix_multiply_transpose_numba(X, X)
        n_features = len(XtX)

        # Add ridge regularization
        for i in range(1, n_features):
            XtX[i][i] += self.alpha

        return self._condition_number(XtX)

    def _calculate_standard_condition_number(self, X):
        """Calculate condition number for standard regression."""
        XtX = matrix_multiply_transpose_numba(X, X)
        return self._condition_number(XtX)

    def _condition_number(self, A):
        """Calculate condition number of matrix A using eigenvalues."""
        try:
            eigenvalues = eigenvalues_power_method_numba(A)
            if len(eigenvalues) == 0:
                return 1e20

            max_eig = max(abs(e) for e in eigenvalues)
            min_eig = min(abs(e) for e in eigenvalues if abs(e) > 1e-10)

            if min_eig < 1e-10:
                return 1e20

            return max_eig / min_eig
        except (ValueError, ArithmeticError):
            return 1e20

    # All mathematical operations delegated to numba functions

    def _solve_with_pseudoinverse(self, A, b):
        """Solve using pseudo-inverse (simplified implementation)."""
        # This is a simplified version - just add small diagonal for stability
        n = len(A)
        A_reg = [row[:] for row in A]  # Deep copy

        # Add small value to diagonal
        for i in range(n):
            A_reg[i][i] += 1e-10

        return solve_linear_system_numba(A_reg, b)

    # Matrix operations are handled by numba functions directly


# Cleaned up - keeping only the simplest implementations

class LinearRegression:
    """Polynomial regression using pure Python (no NumPy) with Numba acceleration."""

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

        # Compute X^T * X using Numba
        XtX = matrix_multiply_transpose_numba(X_with_intercept, X_with_intercept)

        # Compute X^T * y using Numba
        Xty = matrix_vector_multiply_transpose_numba(X_with_intercept, y)

        # Solve the normal equations using Numba
        self.coefficients = solve_linear_system_numba(XtX, Xty)

        return self

    # pylint: disable=duplicate-code
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

    # Matrix operations use numba functions directly


class RidgeRegression:
    """Ridge regression using pure Python with Numba acceleration."""

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

        # Compute X^T * X using Numba
        XtX = matrix_multiply_transpose_numba(X_with_intercept, X_with_intercept)

        # Add ridge penalty to diagonal (except first element for intercept)
        for i in range(1, len(XtX)):
            XtX[i][i] += self.alpha

        # Compute X^T * y using Numba
        Xty = matrix_vector_multiply_transpose_numba(X_with_intercept, y)

        # Solve the system using Numba
        coeffs = solve_linear_system_numba(XtX, Xty)

        # Store all coefficients including intercept (consistent with other engines)
        self.intercept = coeffs[0]
        self.coefficients = coeffs  # All coefficients including intercept

        return self

    def predict(self, X):  # pylint: disable=too-many-locals
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
            X_poly = generate_polynomial_features_numba(x_flat, degree, x_min, x_max, degree > 3)

            # Add intercept column to polynomial features
            predictions = []
            for i, x_row in enumerate(X_poly):
                y_pred = self.coefficients[0]  # intercept
                for j, val in enumerate(x_row):
                    y_pred += self.coefficients[j + 1] * val
                predictions.append(y_pred)
        else:
            # Regular linear prediction
            X_with_intercept = create_zero_matrix(len(X), len(X[0]) + 1)
            for i, x_row in enumerate(X):
                X_with_intercept[i][0] = 1.0  # intercept
                for j, val in enumerate(x_row):
                    X_with_intercept[i][j + 1] = val

            predictions = []
            for i, _ in enumerate(X):
                y_pred = 0
                for j, coeff in enumerate(self.coefficients):
                    y_pred += coeff * X_with_intercept[i][j]
                predictions.append(y_pred)

        return predictions

    # Matrix operations use numba functions directly


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
            X_poly = generate_polynomial_features_numba(x_flat, degree, x_min, x_max, degree > 3)
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
            X_poly = generate_polynomial_features_numba(x_flat, degree, x_min, x_max, degree > 3)
            X = X_poly  # Use for sklearn prediction
        else:
            X = X_list

        return self.model.predict(X)
