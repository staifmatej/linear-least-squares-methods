"""C++ MLPack engine wrapper that mimics Python implementations."""

import numpy as np
import warnings

# Global constants used for bold text and red warning messages.
S_BOLD = "\033[1m"
E_BOLD = "\033[0m"
S_RED = "\033[91m"
E_RED = "\033[0m"

# Try to import C++ module
try:
    import least_squares_cpp
    CPP_AVAILABLE = True
    print(f"{S_BOLD}C++ MLPack engine loaded successfully!{E_BOLD}")
except ImportError as e:
    CPP_AVAILABLE = False
    # Suppress import error message for cleaner user experience
    pass

def fit_error_handling(coefficients):
    """fit error handling for Regression model."""
    if coefficients is None:
        raise ValueError("Model not fitted yet. Call fit() first.")

class LeastSquares:
    """LeastSquares base class that mimics NumPy implementation."""

    def __init__(self, type_regression="LinearRegression"):
        self.type_regression = type_regression
        self.alpha = 1.0
        self.l1_ratio = 0.5
        self.max_iter = 5000
        self.tol = 1e-4
        self.condition_number = None
        self.coefficients = None
        
        # Use NumPy fallback if C++ not available
        if not CPP_AVAILABLE:
            from . import least_squares_numpy
            self._fallback_model = least_squares_numpy.LeastSquares(type_regression)
        else:
            self._fallback_model = None

        types_of_regression = ["LinearRegression", "RidgeRegression", "LassoRegression", "ElasticNetRegression"]
        if type_regression not in types_of_regression:
            raise ValueError(f"Type {self.type_regression} is not a valid predefined type.")

    def multivariate_ols(self, X, Y):
        """
        C++ implementation using MLPack or NumPy fallback.
        X: feature matrix (N, p)
        Y: target variable vector (N, )
        return: w = coefficient vector [w₀, w₁, w₂, ..., wₚ]
        """
        if not CPP_AVAILABLE:
            # Fallback to NumPy implementation
            return self._fallback_model.multivariate_ols(X, Y)
        
        # Use C++ implementation
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_rows, n_cols = X.shape
        if n_rows < n_cols:
            raise ValueError(f"Matrix X must have more rows than columns. Got {n_rows} rows and {n_cols} columns")
        
        # Create appropriate C++ model
        if self.type_regression == "RidgeRegression":
            cpp_model = least_squares_cpp.RidgeRegression(self.alpha)
        else:
            # For Linear regression, use simple linear model
            cpp_model = least_squares_cpp.LinearRegression(1)  # degree=1 for multivariate
        
        # Fit the model
        cpp_model.fit(X.flatten() if X.shape[1] == 1 else X, Y)
        
        # Get coefficients and condition number
        self.coefficients = cpp_model.get_coefficients()
        self.condition_number = cpp_model.get_condition_number()
        
        return self.coefficients

class LinearRegression(LeastSquares):
    """Linear regression using C++ MLPack or NumPy fallback."""

    def __init__(self, degree, type_regression="LinearRegression", normalize=True):
        super().__init__(type_regression="LinearRegression")
        self.degree = degree
        self.normalize = normalize
        self.type_regression = "LinearRegression"
        self.condition_number = None
        self.coefficients = None
        
        if not CPP_AVAILABLE:
            from . import least_squares_numpy
            self._fallback_model = least_squares_numpy.LinearRegression(degree, type_regression, normalize)

    def fit(self, x, y):
        """Fit linear regression model using C++ MLPack."""
        if not CPP_AVAILABLE:
            result = self._fallback_model.fit(x, y)
            self.coefficients = self._fallback_model.coefficients
            self.condition_number = self._fallback_model.condition_number
            return result
        
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        if x.ndim > 1:
            x = x.flatten()
        
        # Create C++ model
        self._cpp_model = least_squares_cpp.LinearRegression(self.degree)
        
        # Fit the model
        self._cpp_model.fit(x, y)
        
        # Store results
        self.coefficients = self._cpp_model.get_coefficients()
        self.condition_number = self._cpp_model.get_condition_number()
        
        return self

    def predict(self, x):
        """Prediction using C++ MLPack model."""
        if not CPP_AVAILABLE:
            return self._fallback_model.predict(x)
        
        fit_error_handling(self.coefficients)
        
        x = np.array(x, dtype=np.float64)
        if x.ndim > 1:
            x = x.flatten()
        
        return self._cpp_model.predict(x)

class RidgeRegression(LeastSquares):
    """Ridge regression using C++ MLPack or NumPy fallback."""

    def __init__(self, alpha=1.0):
        super().__init__(type_regression="RidgeRegression")
        self.alpha = alpha
        self.coefficients = None
        
        if not CPP_AVAILABLE:
            from . import least_squares_numpy
            self._fallback_model = least_squares_numpy.RidgeRegression(alpha)

    def fit(self, x, y):
        """Fit Ridge regression model using C++ MLPack."""
        if not CPP_AVAILABLE:
            result = self._fallback_model.fit(x, y)
            self.coefficients = self._fallback_model.coefficients
            self.condition_number = getattr(self._fallback_model, 'condition_number', None)
            return result
        
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        # Create C++ Ridge model
        self._cpp_model = least_squares_cpp.RidgeRegression(self.alpha)
        
        # Fit the model
        self._cpp_model.fit(x, y)
        
        # Store results
        self.coefficients = self._cpp_model.get_coefficients()
        self.condition_number = self._cpp_model.get_condition_number()
        
        return self

    def predict(self, x):
        """Prediction using C++ Ridge model."""
        if not CPP_AVAILABLE:
            return self._fallback_model.predict(x)
        
        fit_error_handling(self.coefficients)
        
        x = np.array(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        return self._cpp_model.predict(x)

class LassoRegression(LeastSquares):
    """Lasso regression using NumPy fallback (C++ version uses sklearn via NumPy)."""

    def __init__(self, alpha=1.0, max_iter=5000, tol=1e-4):
        super().__init__(type_regression="LassoRegression")
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients = None
        
        # Always use NumPy implementation for Lasso (uses sklearn)
        from . import least_squares_numpy
        self._fallback_model = least_squares_numpy.LassoRegression(alpha, max_iter, tol)

    def fit(self, x, y):
        """Fit Lasso regression model using NumPy/sklearn."""
        result = self._fallback_model.fit(x, y)
        self.coefficients = self._fallback_model.coefficients
        return result

    def predict(self, x):
        """Prediction using Lasso model."""
        return self._fallback_model.predict(x)

class ElasticNetRegression(LeastSquares):
    """ElasticNet regression using NumPy fallback (C++ version uses sklearn via NumPy)."""

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=5000, tol=1e-4):
        super().__init__(type_regression="ElasticNetRegression")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients = None
        
        # Always use NumPy implementation for ElasticNet (uses sklearn)
        from . import least_squares_numpy
        self._fallback_model = least_squares_numpy.ElasticNetRegression(alpha, l1_ratio, max_iter, tol)

    def fit(self, x, y):
        """Fit ElasticNet regression model using NumPy/sklearn."""
        result = self._fallback_model.fit(x, y)
        self.coefficients = self._fallback_model.coefficients
        return result

    def predict(self, x):
        """Prediction using ElasticNet model."""
        return self._fallback_model.predict(x)