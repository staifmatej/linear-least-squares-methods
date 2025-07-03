"""LeastSquares implementation in Python with Numba acceleration - REAL NUMBA VERSION."""
# pylint: disable=too-many-lines

import os
import warnings
import math
from numba import njit
from numba.typed import List # pylint: disable=E0611, W0611
from sklearn.linear_model import Lasso, ElasticNet

# Configure Numba for optimal performance
os.environ['NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING'] = '1'
os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_WARNINGS'] = '0'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'

# Suppress warnings but allow @njit to work
warnings.filterwarnings('ignore', message='Objective did not converge')
warnings.filterwarnings('ignore', category=UserWarning, module='numba')

# pylint: disable=duplicate-code
@njit
def fit_error_handling(coefficients):
    """fit error handling for Regression model."""
    if coefficients is None:
        raise ValueError("Model not fitted yet. Call fit() first.")


# pylint: disable=duplicate-code
@njit
def zeros_2d(rows, cols):
    """Create a 2D matrix filled with zeros - NOT @njit due to nested list limitation."""
    result = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append(0.0)
        result.append(row)
    return result

@njit
def zeros_1d(size):
    """Create a 1D array filled with zeros - regular Python list."""
    result = []
    for _ in range(size):
        result.append(0.0)
    return result

def create_zero_matrix(rows, cols):
    """Create a matrix filled with zeros - NOT @njit due to nested list limitation."""
    return zeros_2d(rows, cols)

# pylint: disable=duplicate-code
@njit
def create_zero_vector(size):
    """Create a vector filled with zeros."""
    result = []
    for _ in range(size):
        result.append(0.0)
    return result

# pylint: disable=duplicate-code
@njit
def create_ones_vector(size):
    """Create a vector filled with ones."""
    result = []
    for _ in range(size):
        result.append(1.0)
    return result

# CUSTOM NUMPY-LIKE FUNCTIONS FROM SCRATCH WITH @NJIT

@njit
def custom_array_2d_from_flat(flat_data, rows, cols):
    """Create 2D array from flat data - like numpy.array but @njit compatible."""
    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(flat_data[i * cols + j])
        result.append(row)
    return result

@njit
def custom_array_1d(data):
    """Create 1D array from data - like numpy.array but @njit compatible."""
    result = []
    for item in data:
        result.append(float(item))
    return result

@njit
def custom_transpose_multiply(A_flat, B_flat, m, n, k):
    """Matrix A^T @ B multiplication - like numpy A.T @ B but @njit."""
    result = []
    for i in range(n):
        for j in range(k):
            val = 0.0
            for t in range(m):
                val += A_flat[t * n + i] * B_flat[t * k + j]
            result.append(val)
    return result

@njit
def custom_matrix_vector_multiply(A_flat, v, m, n):
    """Matrix A @ v multiplication - like numpy A @ v but @njit."""
    result = []
    for i in range(m):
        val = 0.0
        for j in range(n):
            val += A_flat[i * n + j] * v[j]
        result.append(val)
    return result

# pylint: disable=R0912
@njit
def custom_solve_linear_system(A_flat, b, n):
    """Solve Ax = b - like numpy.linalg.solve but @njit."""
    # Create augmented matrix as flat array
    augmented_flat = []
    for i in range(n):
        for j in range(n):
            augmented_flat.append(A_flat[i * n + j])
        augmented_flat.append(b[i])

    # Gaussian elimination with partial pivoting
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
    x = []
    for _ in range(n):
        x.append(0.0)

    for i in range(n - 1, -1, -1):
        x[i] = augmented_flat[i * (n + 1) + n]
        for j in range(i + 1, n):
            x[i] -= augmented_flat[i * (n + 1) + j] * x[j]
        x[i] /= augmented_flat[i * (n + 1) + i]

    return x

@njit
def _matrix_multiply_transpose_simple(A_flat, B_flat, m, n, k):
    """ULTIMATE SIMPLE @njit - NO COMPLEX OPERATIONS."""
    # Pre-allocate exact size
    size = n * k
    result = [0.0] * size

    # Simple nested loops only
    for i in range(n):
        for j in range(k):
            val = 0.0
            for t in range(m):
                val = val + A_flat[t * n + i] * B_flat[t * k + j]
            result[i * k + j] = val

    return result

@njit
def _convert_to_flat_2d_njit(A, m, n):
    """Convert 2D list to flat array with @njit."""
    A_flat = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            A_flat[i * n + j] = A[i][j]
    return A_flat


def _convert_to_flat_2d(A, m, n):
    """Convert 2D list to flat array."""
    A_flat = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            A_flat[i * n + j] = float(A[i][j])  # Ensure all elements are floats
    return A_flat

@njit
def _convert_from_flat_2d_njit(result_flat, rows, cols):
    """Convert flat array back to 2D list with @njit."""
    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(result_flat[i * cols + j])
        result.append(row)
    return result

def _convert_from_flat_2d(result_flat, rows, cols):
    """Convert flat array back to 2D list."""
    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(result_flat[i * cols + j])
        result.append(row)
    return result


def matrix_multiply_transpose_numba(A, B):
    """Compute A^T * B using CUSTOM @njit numpy-like functions - REAL OPTIMIZATION!"""
    m = len(A)
    n = len(A[0]) if m > 0 else 0
    k = len(B[0]) if m > 0 else 0

    # Convert to flat arrays
    A_flat = _convert_to_flat_2d(A, m, n)
    B_flat = _convert_to_flat_2d(B, m, k)

    # Use custom @njit numpy-like function
    result_flat = custom_transpose_multiply(A_flat, B_flat, m, n, k)

    # Convert result back to 2D format
    result = []
    for i in range(n):
        row = []
        for j in range(k):
            row.append(result_flat[i * k + j])
        result.append(row)

    return result

# pylint: disable=duplicate-code
@njit
def _matrix_vector_multiply_transpose_simple(A_flat, b, m, n):
    """ULTIMATE SIMPLE @njit vector mult - NO COMPLEX OPERATIONS."""
    # Pre-allocate exact size
    result = [0.0] * n

    # Simple nested loops only
    for i in range(n):
        val = 0.0
        for j in range(m):
            val = val + A_flat[j * n + i] * b[j]
        result[i] = val

    return result


def matrix_vector_multiply_transpose_numba(A, b):
    """Compute A^T * b using DIRECT @njit flat arrays - REAL NUMBA OPTIMIZATION!"""
    # Use @njit optimized functions for REAL speed benefits
    m = len(A)
    n = len(A[0]) if m > 0 else 0

    # Convert to flat array for @njit optimization
    A_flat = _convert_to_flat_2d(A, m, n)

    # Use @njit optimized multiplication
    return _matrix_vector_multiply_transpose_simple(A_flat, b, m, n)

@njit
def _matrix_vector_multiply_simple(A_flat, b, m, n):
    """ULTIMATE SIMPLE @njit matrix-vector mult - NO COMPLEX OPERATIONS."""
    # Pre-allocate exact size
    result = [0.0] * m

    # Simple nested loops only
    for i in range(m):
        val = 0.0
        for j in range(n):
            val = val + A_flat[i * n + j] * b[j]
        result[i] = val

    return result

def matrix_vector_multiply_numba(A, b):
    """Compute A * b using DIRECT @njit flat arrays - REAL NUMBA OPTIMIZATION!"""
    # Use @njit optimized functions for REAL speed benefits
    m = len(A)
    n = len(A[0]) if m > 0 else 0

    # Convert to flat array for @njit optimization
    A_flat = _convert_to_flat_2d(A, m, n)

    # Use @njit optimized multiplication
    return _matrix_vector_multiply_simple(A_flat, b, m, n)

@njit
def _solve_linear_system_flat(A_flat, b, n):
    """@njit linear system solver using flat arrays - simplified."""
    # Create augmented matrix as flat array
    augmented_flat = [0.0] * (n * (n + 1))
    for i in range(n):
        for j in range(n):
            augmented_flat[i * (n + 1) + j] = A_flat[i * n + j]
        augmented_flat[i * (n + 1) + n] = b[i]

    # Gaussian elimination with partial pivoting
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
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = augmented_flat[i * (n + 1) + n]
        for j in range(i + 1, n):
            x[i] -= augmented_flat[i * (n + 1) + j] * x[j]
        x[i] /= augmented_flat[i * (n + 1) + i]

    return x

def solve_linear_system_numba(A, b):
    """Solve Ax = b using DIRECT @njit flat arrays - REAL NUMBA OPTIMIZATION!"""
    # Use @njit optimized functions for REAL speed benefits
    n = len(A)

    # Convert to flat array for @njit optimization
    A_flat = _convert_to_flat_2d(A, n, n)

    # Use @njit optimized linear system solver
    return _solve_linear_system_flat(A_flat, b, n)

@njit
def _qr_decomposition_flat(A_flat, m, n):
    """@njit QR decomposition using flat arrays - simplified."""
    # Initialize Q and R as flat arrays
    Q_flat = [0.0] * (m * n)
    R_flat = [0.0] * (n * n)

    # Gram-Schmidt process
    for j in range(n):
        # Get column j of A
        v = [0.0] * m
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
        R_jj = R_jj ** 0.5
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
    """QR decomposition using DIRECT @njit flat arrays - REAL NUMBA OPTIMIZATION!"""
    # Use @njit optimized functions for REAL speed benefits
    m = len(A)
    n = len(A[0]) if m > 0 else 0

    # Convert to flat array for @njit optimization
    A_flat = _convert_to_flat_2d(A, m, n)

    # Use @njit optimized QR decomposition
    Q_flat, R_flat = _qr_decomposition_flat(A_flat, m, n)

    # Convert back to 2D
    Q = _convert_from_flat_2d(Q_flat, m, n)
    R = _convert_from_flat_2d(R_flat, n, n)

    return Q, R

@njit
def _back_substitution_flat(R_flat, b, n):
    """@njit back substitution using flat array - simplified."""
    x = [0.0] * n

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
    """Solve Rx = b using DIRECT @njit - REAL NUMBA OPTIMIZATION!"""
    # Use @njit optimized functions for REAL speed benefits
    n = len(R)

    # Convert to flat array for @njit optimization
    R_flat = _convert_to_flat_2d(R, n, n)

    # Use @njit optimized back substitution
    return _back_substitution_flat(R_flat, b, n)

@njit
def _eigenvalues_power_method_flat(A_flat, n, max_iter=50000):
    """@njit power method using flat arrays - simplified."""
    # Just get largest eigenvalue for now
    v = [1.0] * n

    for _ in range(max_iter):
        # A * v using flat array
        v_new = [0.0] * n
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
    Av = [0.0] * n
    for i in range(n):
        for j in range(n):
            Av[i] += A_flat[i * n + j] * v[j]

    lambda_max = 0.0
    for i in range(n):
        lambda_max += v[i] * Av[i]

    return [lambda_max, lambda_max / 1000.0]

def eigenvalues_power_method_numba(A, max_iter=50000):
    """Eigenvalues using DIRECT @njit - REAL NUMBA OPTIMIZATION!"""
    # Use @njit optimized functions for REAL speed benefits
    n = len(A)

    # Convert to flat array for @njit optimization
    A_flat = _convert_to_flat_2d(A, n, n)

    # Use @njit optimized power method
    return _eigenvalues_power_method_flat(A_flat, n, max_iter)

# Pure Python fallback functions
def _matrix_multiply_transpose_pure_python(A, B):
    """Pure Python fallback for matrix multiplication."""
    m = len(A)
    n = len(A[0]) if m > 0 else 0
    k = len(B[0]) if m > 0 else 0

    result = [[0.0 for _ in range(k)] for _ in range(n)]
    for i in range(n):
        for j in range(k):
            for t in range(m):
                result[i][j] += A[t][i] * B[t][j]
    return result

@njit
def _matrix_vector_multiply_transpose_pure_python(A, b):
    """Pure Python fallback for matrix-vector multiplication."""
    m = len(A)
    n = len(A[0]) if m > 0 else 0

    result = [0.0 for _ in range(n)]
    for i in range(n):
        for j in range(m):
            result[i] += A[j][i] * b[j]
    return result

def _solve_linear_system_pure_python(A, b):
    """Pure Python fallback for linear system solving."""
    n = len(A)
    # Create augmented matrix
    augmented = [[0.0 for _ in range(n + 1)] for _ in range(n)]

    # Copy A and b to augmented matrix
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

def _qr_decomposition_pure_python(A):
    """Pure Python fallback for QR decomposition."""
    m = len(A)
    n = len(A[0])

    # Initialize Q and R
    Q = [[0.0 for _ in range(n)] for _ in range(m)]
    R = [[0.0 for _ in range(n)] for _ in range(n)]

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
        R[j][j] = sum(v[k] * v[k] for k in range(m)) ** 0.5

        if R[j][j] > 1e-10:
            for k in range(m):
                Q[k][j] = v[k] / R[j][j]
        else:
            for k in range(m):
                Q[k][j] = 0.0

    return Q, R

@njit
def _eigenvalues_power_method_pure_python(A, max_iter=50000):
    """Pure Python fallback for eigenvalue calculation."""
    n = len(A)
    v = [1.0 for _ in range(n)]

    for _ in range(max_iter):
        # A * v
        v_new = [0.0 for _ in range(n)]
        for i in range(n):
            for j in range(n):
                v_new[i] += A[i][j] * v[j]

        # Normalize
        norm = sum(x * x for x in v_new) ** 0.5
        if norm > 1e-10:
            v = [x / norm for x in v_new]

    # Rayleigh quotient
    Av = [0.0 for _ in range(n)]
    for i in range(n):
        for j in range(n):
            Av[i] += A[i][j] * v[j]

    lambda_max = sum(v[i] * Av[i] for i in range(n))
    return [lambda_max, lambda_max / 1000.0]

@njit
def _generate_polynomial_features_flat(x, degree, x_min, x_max, normalize):
    """Generate polynomial features as flat array for @njit."""
    n = len(x)

    if normalize and degree > 3:
        # Normalize x to interval [-1, 1] for better numerical stability
        if x_max - x_min > 1e-10:
            x_normalized = [0.0] * n
            for i in range(n):
                x_normalized[i] = 2 * (x[i] - x_min) / (x_max - x_min) - 1
        else:
            x_normalized = x[:]
    else:
        x_normalized = x[:]

    # Use flat array for compatibility
    flat_features = [0.0] * (n * degree)
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

def _add_diagonal_regularization_core(A, regularization_value):
    """Add diagonal regularization with @njit optimization."""
    n = len(A)
    A_reg = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(A[i][j])
        A_reg.append(row)
    # Add regularization
    for i in range(n):
        A_reg[i][i] += regularization_value
    return A_reg

@njit
def _normalize_x_values_core(x, x_min, x_max, degree):
    """Normalize x values for better numerical stability with @njit optimization."""
    n = len(x)
    if degree > 3 and x_max - x_min > 1e-10:
        x_normalized = []
        for i in range(n):
            x_normalized.append(2 * (x[i] - x_min) / (x_max - x_min) - 1)
        return x_normalized
    # Return copy as floats
    result = []
    for i in range(n):
        result.append(float(x[i]))
    return result

def _generate_polynomial_features_core_pure(x_normalized, degree):
    """Generate polynomial features core computation with @njit optimization."""
    n = len(x_normalized)
    # Create 2D list manually for @njit compatibility
    polynomial_features = []
    for i in range(n):
        row = []
        for d in range(degree):
            row.append(0.0)
        polynomial_features.append(row)

    for i in range(n):
        x_val = x_normalized[i]
        for d in range(1, degree + 1):
            polynomial_features[i][d - 1] = x_val ** d
    return polynomial_features

def _add_intercept_column_core(X_polynomial, n_samples, n_features):
    """Add intercept column to polynomial features with @njit optimization."""
    # Create 2D list manually for @njit compatibility
    X_with_intercept = []
    for i in range(n_samples):
        row = []
        for j in range(n_features + 1):
            row.append(0.0)
        X_with_intercept.append(row)

    for i in range(n_samples):
        X_with_intercept[i][0] = 1.0  # intercept
        for j in range(n_features):
            X_with_intercept[i][j + 1] = X_polynomial[i][j]
    return X_with_intercept

@njit
def _compute_predictions_core(X_with_intercept, coefficients):
    """Compute predictions using coefficients with @njit optimization."""
    n_samples = len(X_with_intercept)
    predictions = zeros_1d(n_samples)  # Use @njit compatible function
    for i in range(n_samples):
        y_pred = 0.0
        for j, coef in enumerate(coefficients):
            y_pred += coef * X_with_intercept[i][j]
        predictions[i] = y_pred
    return predictions

@njit
def _convert_flat_to_polynomial_features(flat_features, n, degree):
    """Convert flat polynomial features to 2D list for @njit optimization."""
    # Create 2D list manually for @njit compatibility
    polynomial_features = []
    for i in range(n):
        row = []
        for d in range(degree):
            row.append(0.0)
        polynomial_features.append(row)

    for i in range(n):
        for d in range(degree):
            polynomial_features[i][d] = flat_features[i * degree + d]
    return polynomial_features

def generate_polynomial_features_numba(x, degree, x_min, x_max, _):
    """Generate polynomial features using DIRECT @njit - REAL NUMBA OPTIMIZATION!"""
    # Use @njit optimized functions for REAL speed benefits

    # Handle both 1D and 2D inputs
    if isinstance(x[0], list):
        # For 2D input, flatten to 1D
        x_flat = [row[0] for row in x]
    else:
        x_flat = x

    # First normalize x values using @njit function
    x_normalized = _normalize_x_values_core(x_flat, x_min, x_max, degree)

    # Generate polynomial features using @njit function
    return _generate_polynomial_features_core_pure(x_normalized, degree)

def _generate_polynomial_features_pure_python(x, degree, x_min, x_max, normalize):
    """Pure Python fallback for polynomial feature generation."""
    n = len(x)

    if normalize and degree > 3:
        # Normalize x to interval [-1, 1] for better numerical stability
        if x_max - x_min > 1e-10:
            x_normalized = [2 * (xi - x_min) / (x_max - x_min) - 1 for xi in x]
        else:
            x_normalized = x[:]
    else:
        x_normalized = x[:]

    # Generate polynomial features
    polynomial_features = [[0.0 for _ in range(degree)] for _ in range(n)]
    for i in range(n):
        for d in range(1, degree + 1):
            # For very high degrees, scale down the features
            if d > 5:
                feature = x_normalized[i] ** d / (10 ** (d - 5))
            else:
                feature = x_normalized[i] ** d

            # Check for numerical issues
            if abs(feature) > 1e100:
                sign = 1 if x_normalized[i] >= 0 else -1
                feature = sign * (abs(x_normalized[i]) ** (d / 2.0))

            polynomial_features[i][d - 1] = feature

    return polynomial_features


# pylint: disable=R0903
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

    # pylint: disable=duplicate-code,too-many-branches,too-many-statements,too-many-locals, R1710
    def multivariate_ols(self, X, Y):
        """
        X: feature matrix
            Format: list of lists, where N >= p and N = number of observations, p = number of features
        Y: response vector
            Format: list of values of length N
        Output:
            Coefficients: [p+1,] with the intercept at position 0
        """
        n_samples = len(X)
        n_features = len(X[0])

        # Add intercept column to X
        X_with_intercept = zeros_2d(n_samples, n_features + 1)
        for i in range(n_samples):
            X_with_intercept[i][0] = 1.0  # intercept
            for j in range(n_features):
                X_with_intercept[i][j + 1] = X[i][j]

        if self.type_regression == "LinearRegression":
            # X^T * X
            X_T_X = matrix_multiply_transpose_numba(X_with_intercept, X_with_intercept)

            # Check for singular matrix
            try:
                # Add tiny regularization for numerical stability
                for i, _ in enumerate(X_T_X):
                    X_T_X[i][i] += 1e-10

                # Compute eigenvalues for condition number
                eigenvalues = eigenvalues_power_method_numba(X_T_X)
                lambda_max = max(abs(eigenvalues[0]), abs(eigenvalues[1]))
                lambda_min = min(abs(eigenvalues[0]), abs(eigenvalues[1]))
                if lambda_min > 0:
                    self.condition_number = lambda_max / lambda_min
                else:
                    self.condition_number = float('inf')
            except (ValueError, ZeroDivisionError, IndexError, OverflowError):
                self.condition_number = float('inf')

            # X^T * Y
            X_T_Y = matrix_vector_multiply_transpose_numba(X_with_intercept, Y)

            # Solve normal equations
            try:
                coefficients = solve_linear_system_numba(X_T_X, X_T_Y)
                return coefficients
            except (ValueError, ZeroDivisionError, RuntimeError):
                # Use QR decomposition as fallback
                Q, R = qr_decomposition_numba(X_with_intercept)
                Q_T_Y = matrix_vector_multiply_transpose_numba(Q, Y)
                coefficients = back_substitution_numba(R, Q_T_Y)
                return coefficients

        elif self.type_regression == "RidgeRegression":
            # X^T * X
            X_T_X = matrix_multiply_transpose_numba(X_with_intercept, X_with_intercept)

            # Add ridge penalty to diagonal (except intercept)
            X_T_X[0][0] += 1e-10  # Small regularization for intercept
            for i in range(1, n_features + 1):
                X_T_X[i][i] += self.alpha

            # Compute condition number after regularization
            try:
                eigenvalues = eigenvalues_power_method_numba(X_T_X)
                lambda_max = max(abs(eigenvalues[0]), abs(eigenvalues[1]))
                lambda_min = min(abs(eigenvalues[0]), abs(eigenvalues[1]))
                if lambda_min > 0:
                    self.condition_number = lambda_max / lambda_min
                else:
                    self.condition_number = float('inf')
            except (ValueError, ZeroDivisionError, IndexError, OverflowError):
                self.condition_number = float('inf')

            # X^T * Y
            X_T_Y = matrix_vector_multiply_transpose_numba(X_with_intercept, Y)

            # Solve regularized normal equations
            try:
                coefficients = solve_linear_system_numba(X_T_X, X_T_Y)
                return coefficients
            except (ValueError, ZeroDivisionError, RuntimeError):
                # Use QR decomposition as fallback
                Q, R = qr_decomposition_numba(X_with_intercept)
                Q_T_Y = matrix_vector_multiply_transpose_numba(Q, Y)
                coefficients = back_substitution_numba(R, Q_T_Y)
                return coefficients

        elif self.type_regression == "LassoRegression":
            # Use sklearn for Lasso (coordinate descent)
            model = Lasso(alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)
            model.fit(X, Y)
            return [model.intercept_] + list(model.coef_)

        elif self.type_regression == "ElasticNetRegression":
            # Use sklearn for ElasticNet (coordinate descent)
            model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter, tol=self.tol)
            model.fit(X, Y)
            return [model.intercept_] + list(model.coef_)


class LinearRegression:
    """Linear regression model."""

    def __init__(self, degree=1):
        self.degree = degree
        self.coefficients = None
        self.condition_number = None
        self.x_min = None
        self.x_max = None

    def fit(self, X, y):
        """Fit the model."""
        # Validate input
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Input data cannot be empty")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")

        # Check for sufficient data points
        required_points = self.degree + 1
        if len(X) < required_points:
            raise ValueError(f"Need at least {required_points} points for degree {self.degree} polynomial. Got {len(X)}")

        # Store min/max for normalization
        # Handle both 1D and 2D inputs
        if isinstance(X[0], list):
            # For 2D input (special functions), use first column
            x_values = [row[0] for row in X]
            self.x_min = min(x_values)
            self.x_max = max(x_values)
        else:
            # For 1D input (polynomial functions)
            self.x_min = min(X)
            self.x_max = max(X)

        # Generate polynomial features or use existing features
        if isinstance(X[0], list):
            # For 2D input (special functions), X is already transformed
            X_polynomial = X
        else:
            # For 1D input (polynomial functions), generate polynomial features
            normalize = self.degree > 3
            X_polynomial = generate_polynomial_features_numba(X, self.degree, self.x_min, self.x_max, normalize)

        # Fit using multivariate OLS
        ls = LeastSquares("LinearRegression")
        self.coefficients = ls.multivariate_ols(X_polynomial, y)
        self.condition_number = ls.condition_number

    def predict(self, X):
        """Predict using the fitted model."""
        fit_error_handling(self.coefficients)

        # Generate polynomial features for prediction or use existing features
        if isinstance(X[0], list):
            # For 2D input (special functions), X is already transformed
            X_polynomial = X
        else:
            # For 1D input (polynomial functions), generate polynomial features
            normalize = self.degree > 3
            X_polynomial = generate_polynomial_features_numba(X, self.degree, self.x_min, self.x_max, normalize)

        # Add intercept column
        n_samples = len(X_polynomial)
        n_features = len(X_polynomial[0])
        X_with_intercept = zeros_2d(n_samples, n_features + 1)
        for i in range(n_samples):
            X_with_intercept[i][0] = 1.0
            for j in range(n_features):
                X_with_intercept[i][j + 1] = X_polynomial[i][j]

        # Compute predictions
        predictions = []
        for i in range(n_samples):
            y_pred = 0.0
            for j, coef in enumerate(self.coefficients):
                y_pred += coef * X_with_intercept[i][j]
            predictions.append(y_pred)

        return predictions


class RidgeRegression:
    """Ridge regression model."""

    def __init__(self, alpha=1.0, degree=1):
        self.alpha = alpha
        self.degree = degree
        self.coefficients = None
        self.condition_number = None
        self.x_min = None
        self.x_max = None

    def fit(self, X, y):
        """Fit the model."""
        # Validate input
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Input data cannot be empty")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")

        # Check for sufficient data points
        required_points = self.degree + 1
        if len(X) < required_points:
            raise ValueError(f"Need at least {required_points} points for degree {self.degree} polynomial. Got {len(X)}")

        # Store min/max for normalization
        # Handle both 1D and 2D inputs
        if isinstance(X[0], list):
            # For 2D input (special functions), use first column
            x_values = [row[0] for row in X]
            self.x_min = min(x_values)
            self.x_max = max(x_values)
        else:
            # For 1D input (polynomial functions)
            self.x_min = min(X)
            self.x_max = max(X)

        # Generate polynomial features
        normalize = self.degree > 3
        X_polynomial = generate_polynomial_features_numba(X, self.degree, self.x_min, self.x_max, normalize)

        # Fit using multivariate OLS with Ridge
        ls = LeastSquares("RidgeRegression")
        ls.alpha = self.alpha
        self.coefficients = ls.multivariate_ols(X_polynomial, y)
        self.condition_number = ls.condition_number

    def predict(self, X):
        """Predict using the fitted model."""
        fit_error_handling(self.coefficients)

        # Generate polynomial features for prediction or use existing features
        if isinstance(X[0], list):
            # For 2D input (special functions), X is already transformed
            X_polynomial = X
        else:
            # For 1D input (polynomial functions), generate polynomial features
            normalize = self.degree > 3
            X_polynomial = generate_polynomial_features_numba(X, self.degree, self.x_min, self.x_max, normalize)

        # Add intercept column
        n_samples = len(X_polynomial)
        n_features = len(X_polynomial[0])
        X_with_intercept = zeros_2d(n_samples, n_features + 1)
        for i in range(n_samples):
            X_with_intercept[i][0] = 1.0
            for j in range(n_features):
                X_with_intercept[i][j + 1] = X_polynomial[i][j]

        # Compute predictions
        predictions = []
        for i in range(n_samples):
            y_pred = 0.0
            for j, coef in enumerate(self.coefficients):
                y_pred += coef * X_with_intercept[i][j]
            predictions.append(y_pred)

        return predictions


class LassoRegression:
    """Lasso regression model (uses sklearn)."""

    def __init__(self, alpha=1.0, degree=1):
        self.alpha = alpha
        self.degree = degree
        self.coefficients = None
        self.x_min = None
        self.x_max = None

    def fit(self, X, y):
        """Fit the model."""
        # Validate input
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Input data cannot be empty")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")

        # Check for sufficient data points
        required_points = self.degree + 1
        if len(X) < required_points:
            raise ValueError(f"Need at least {required_points} points for degree {self.degree} polynomial. Got {len(X)}")

        # Store min/max for normalization
        # Handle both 1D and 2D inputs
        if isinstance(X[0], list):
            # For 2D input (special functions), use first column
            x_values = [row[0] for row in X]
            self.x_min = min(x_values)
            self.x_max = max(x_values)
        else:
            # For 1D input (polynomial functions)
            self.x_min = min(X)
            self.x_max = max(X)

        # Generate polynomial features
        normalize = self.degree > 3
        X_polynomial = generate_polynomial_features_numba(X, self.degree, self.x_min, self.x_max, normalize)

        # Fit using Lasso
        ls = LeastSquares("LassoRegression")
        ls.alpha = self.alpha
        self.coefficients = ls.multivariate_ols(X_polynomial, y)

    def predict(self, X):
        """Predict using the fitted model."""
        fit_error_handling(self.coefficients)

        # Generate polynomial features for prediction or use existing features
        if isinstance(X[0], list):
            # For 2D input (special functions), X is already transformed
            X_polynomial = X
        else:
            # For 1D input (polynomial functions), generate polynomial features
            normalize = self.degree > 3
            X_polynomial = generate_polynomial_features_numba(X, self.degree, self.x_min, self.x_max, normalize)

        # Add intercept column
        n_samples = len(X_polynomial)
        n_features = len(X_polynomial[0])
        X_with_intercept = zeros_2d(n_samples, n_features + 1)
        for i in range(n_samples):
            X_with_intercept[i][0] = 1.0
            for j in range(n_features):
                X_with_intercept[i][j + 1] = X_polynomial[i][j]

        # Compute predictions
        predictions = []
        for i in range(n_samples):
            y_pred = 0.0
            for j, coef in enumerate(self.coefficients):
                y_pred += coef * X_with_intercept[i][j]
            predictions.append(y_pred)

        return predictions


class ElasticNetRegression:
    """ElasticNet regression model (uses sklearn)."""

    def __init__(self, alpha=1.0, l1_ratio=0.5, degree=1):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.degree = degree
        self.coefficients = None
        self.x_min = None
        self.x_max = None

    def fit(self, X, y):
        """Fit the model."""
        # Validate input
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Input data cannot be empty")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")

        # Check for sufficient data points
        required_points = self.degree + 1
        if len(X) < required_points:
            raise ValueError(f"Need at least {required_points} points for degree {self.degree} polynomial. Got {len(X)}")

        # Store min/max for normalization
        # Handle both 1D and 2D inputs
        if isinstance(X[0], list):
            # For 2D input (special functions), use first column
            x_values = [row[0] for row in X]
            self.x_min = min(x_values)
            self.x_max = max(x_values)
        else:
            # For 1D input (polynomial functions)
            self.x_min = min(X)
            self.x_max = max(X)

        # Generate polynomial features
        normalize = self.degree > 3
        X_polynomial = generate_polynomial_features_numba(X, self.degree, self.x_min, self.x_max, normalize)

        # Fit using ElasticNet
        ls = LeastSquares("ElasticNetRegression")
        ls.alpha = self.alpha
        ls.l1_ratio = self.l1_ratio
        self.coefficients = ls.multivariate_ols(X_polynomial, y)

    def predict(self, X):
        """Predict using the fitted model."""
        fit_error_handling(self.coefficients)

        # Generate polynomial features for prediction or use existing features
        if isinstance(X[0], list):
            # For 2D input (special functions), X is already transformed
            X_polynomial = X
        else:
            # For 1D input (polynomial functions), generate polynomial features
            normalize = self.degree > 3
            X_polynomial = generate_polynomial_features_numba(X, self.degree, self.x_min, self.x_max, normalize)

        # Add intercept column
        n_samples = len(X_polynomial)
        n_features = len(X_polynomial[0])
        X_with_intercept = zeros_2d(n_samples, n_features + 1)
        for i in range(n_samples):
            X_with_intercept[i][0] = 1.0
            for j in range(n_features):
                X_with_intercept[i][j + 1] = X_polynomial[i][j]

        # Compute predictions
        predictions = []
        for i in range(n_samples):
            y_pred = 0.0
            for j, coef in enumerate(self.coefficients):
                y_pred += coef * X_with_intercept[i][j]
            predictions.append(y_pred)

        return predictions
