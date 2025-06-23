"""Unified regression engine for multiple implementations and regression types."""

import numpy as np
from approaches.least_squares_numpy import (
    PolynomialRegression, RidgeRegression,
    LassoRegression, ElasticNetRegression
)

# Global constants used for bold text and red warning messages.
S_BOLD = "\033[1m"
E_BOLD = "\033[0m"
S_RED = "\033[91m"
E_RED = "\033[0m"

class RegressionRun:
    """Unified regression engine that supports multiple implementations and regression types."""

    def __init__(self, engine_choice, regression_types, function_types):
        self.engine_choice = engine_choice
        self.regression_types = regression_types
        self.function_types = function_types
        self.results = {}

        self.engine_mapping = {
            1: "cpp",
            2: "numpy",
            3: "numba",
            4: "pure"
        }

        self.regression_mapping = {
            1: "Polynomial",
            2: "Ridge",
            3: "Lasso",
            4: "ElasticNet"
        }

        self.function_degree_mapping = {
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7
        }

    def run_regressions(self, X, y):
        """Run all selected regression types with all selected function types."""

        for reg_type in self.regression_types:
            for func_type in self.function_types:
                try:
                    result = self._run_single_regression(X, y, reg_type, func_type)
                    self.results[(reg_type, func_type)] = result
                except (ValueError, RuntimeError, np.linalg.LinAlgError, KeyError):
                    self.results[(reg_type, func_type)] = None

        return self.results

    def _run_single_regression(self, X, y, regression_type, function_type):
        """Run single regression based on engine choice."""
        if self.engine_choice == 2:
            return self._run_numpy_regression(X, y, regression_type, function_type)
        if self.engine_choice == 1:
            return self._run_cpp_regression(X, y, regression_type, function_type)
        if self.engine_choice == 3:
            return self._run_numba_regression(X, y, regression_type, function_type)
        if self.engine_choice == 4:
            return self._run_pure_regression(X, y, regression_type, function_type)

        raise ValueError(f"Unknown engine choice: {self.engine_choice}")

    # pylint: disable=duplicate-code,too-many-return-statements
    def _transform_features_for_function(self, X, y, function_type):
        """Transform features according to function type."""
        X = X.flatten()

        # Handle negative and zero values
        min_val = 1e-10

        if function_type == 8:  # Log-Linear: y = a + b*log(x)
            X_positive = np.where(X > 0, X, min_val)
            X_transformed = np.log(X_positive).reshape(-1, 1)
            return X_transformed, y

        if function_type == 9:  # Log-Polynomial: y = a + b*log(x) + c*log(x)^2
            X_positive = np.where(X > 0, X, min_val)
            X_log = np.log(X_positive)
            X_transformed = np.column_stack([X_log, X_log ** 2])
            return X_transformed, y

        if function_type == 10:  # Semi-Log: log(y) = a + b*x
            y_positive = np.where(y > 0, y, min_val)
            y_transformed = np.log(y_positive)
            return X.reshape(-1, 1), y_transformed

        if function_type == 11:  # Square Root: y = a + b*sqrt(x)
            X_positive = np.where(X >= 0, X, 0)
            X_transformed = np.sqrt(X_positive).reshape(-1, 1)
            return X_transformed, y

        if function_type == 12:  # Inverse: y = a + b/x
            X_nonzero = np.where(np.abs(X) > min_val, X, min_val * np.sign(X))
            X_nonzero = np.where(X_nonzero == 0, min_val, X_nonzero)
            X_transformed = (1.0 / X_nonzero).reshape(-1, 1)
            return X_transformed, y

        if function_type == 13:  # Log-Sqrt: y = a + b*log(x) + c*sqrt(x)
            X_positive = np.where(X > 0, X, min_val)
            X_log = np.log(X_positive)
            X_sqrt = np.sqrt(X_positive)
            X_transformed = np.column_stack([X_log, X_sqrt])
            return X_transformed, y

        if function_type == 14:  # Mixed: y = a + b*x + c*log(x)
            X_positive = np.where(X > 0, X, min_val)
            X_log = np.log(X_positive)
            X_transformed = np.column_stack([X, X_log])
            return X_transformed, y

        if function_type == 15:  # Poly-Log: y = a + b*x + c*x^2 + d*log(x)
            X_positive = np.where(X > 0, X, min_val)
            X_log = np.log(X_positive)
            X_transformed = np.column_stack([X, X ** 2, X_log])
            return X_transformed, y

        if function_type == 16:  # Volatility Mix: y = a + b*sqrt(x) + c/x
            X_positive = np.where(X > 0, X, min_val)
            X_sqrt = np.sqrt(X_positive)
            X_inv = 1.0 / X_positive
            X_transformed = np.column_stack([X_sqrt, X_inv])
            return X_transformed, y

        raise ValueError(f"Unknown function type: {function_type}")

    def _get_regression_model(self, regression_type, **kwargs):
        """Return model instance based on regression type."""
        if regression_type == 1:
            return PolynomialRegression(degree=1, **kwargs)
        if regression_type == 2:
            return RidgeRegression(alpha=1.0)
        if regression_type == 3:
            return LassoRegression(alpha=1.0)
        if regression_type == 4:
            return ElasticNetRegression(alpha=1.0, l1_ratio=0.5)
        raise ValueError(f"Unknown regression type: {regression_type}")

    def _run_numpy_regression(self, X, y, regression_type, function_type):
        """Run regression using NumPy implementation - FIXED VERSION."""
        if 1 <= function_type <= 7:
            # For polynomial functions
            degree = self.function_degree_mapping[function_type]

            if regression_type == 1:  # Polynomial regression
                model = PolynomialRegression(degree=degree)
                model.fit(X.flatten(), y)
                coeffs = model.coefficients
                return {
                    'model': model,
                    'coefficients': coeffs,
                    'degree': degree,
                    'regression_type': 'Polynomial',
                    'function_type': function_type
                }

            # Ridge, Lasso, ElasticNet for polynomials
            # Create polynomial features
            X_poly = self._generate_polynomial_features(X, degree)

            if regression_type == 2:
                model = RidgeRegression(alpha=1.0)
            elif regression_type == 3:
                model = LassoRegression(alpha=1.0)
            elif regression_type == 4:
                model = ElasticNetRegression(alpha=1.0, l1_ratio=0.5)

            model.fit(X_poly, y)
            coeffs = model.coefficients

            return {
                'model': model,
                'coefficients': coeffs,
                'degree': degree,
                'regression_type': self.regression_mapping[regression_type],
                'function_type': function_type
            }

        # For special functions 8-16
        # FIXED: For polynomial regression on special functions, we don't use polynomial degree
        if regression_type == 1:  # Polynomial regression
            # For special functions, we just do linear regression on transformed features
            X_transformed, y_transformed = self._transform_features_for_function(X, y, function_type)

            # Create a simple linear model on transformed features
            model = RidgeRegression(alpha=0.0)  # alpha=0 means regular least squares
            model.fit(X_transformed, y_transformed)

            return {
                'model': model,
                'coefficients': model.coefficients,
                'function_type': function_type,
                'regression_type': 'Polynomial',
                'transformation': f'Function {function_type}',
                'is_transformed': True
            }

        # For Ridge, Lasso, ElasticNet on special functions
        X_transformed, y_transformed = self._transform_features_for_function(X, y, function_type)

        # Get model based on regression type
        model = self._get_regression_model(regression_type)

        # Fit model
        model.fit(X_transformed, y_transformed)

        return {
            'model': model,
            'coefficients': model.coefficients,
            'function_type': function_type,
            'regression_type': self.regression_mapping[regression_type],
            'transformation': f'Function {function_type}',
            'is_transformed': True
        }

    def _generate_polynomial_features(self, X, degree):
        """Generate polynomial features up to specified degree with better normalization."""
        X_flat = X.flatten() if X.ndim > 1 else X

        # Enhanced normalization for high-degree polynomials
        if degree > 3:
            x_min, x_max = X_flat.min(), X_flat.max()
            if x_max - x_min > 1e-10:
                # Normalize to [-1, 1] range
                X_normalized = 2 * (X_flat - x_min) / (x_max - x_min) - 1
            else:
                X_normalized = X_flat
        else:
            X_normalized = X_flat

        polynomial_features = []
        for d in range(1, degree + 1):
            # For very high degrees, scale down the features
            if d > 5:
                feature = X_normalized ** d / (10 ** (d - 5))
            else:
                feature = X_normalized ** d
            polynomial_features.append(feature)

        return np.column_stack(polynomial_features)

    def _run_cpp_regression(self, _X, _y, _regression_type, _function_type):
        """Run regression using C++ implementation."""
        return {
            'status': 'not_implemented',
            'engine': 'cpp',
            'message': 'C++ implementation pending'
        }

    def _run_numba_regression(self, _X, _y, _regression_type, _function_type):
        """Run regression using Numba implementation."""
        return {
            'status': 'not_implemented',
            'engine': 'numba',
            'message': 'Numba implementation pending'
        }

    def _run_pure_regression(self, _X, _y, _regression_type, _function_type):
        """Run regression using Pure Python implementation."""
        return {
            'status': 'not_implemented',
            'engine': 'pure',
            'message': 'Pure Python implementation pending'
        }

    def print_results(self):
        """Print summary of all regression results."""

        successful = 0
        failed = 0
        not_implemented = 0

        for _, result in self.results.items():
            if result is None:
                failed += 1
            elif result.get('status') == 'not_implemented':
                not_implemented += 1
            else:
                successful += 1

        print(f"\n{S_BOLD}Summary:{E_BOLD} {successful} successful, {failed} failed, {not_implemented} not implemented")
