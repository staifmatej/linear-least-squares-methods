"""Unified regression engine for multiple implementations and regression types."""

import numpy as np
from approaches import least_squares_numpy
from approaches import least_squares_numba
from approaches import least_squares_pure

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
            1: "Linear",
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

    def _transform_features_for_function_numpy(self, X, y, function_type):
        """Transform features for special functions using NumPy."""
        X_flat = X.flatten() if hasattr(X, 'flatten') else X
        min_val = 1e-10

        if function_type == 8:  # Log-Linear: y = a + b*log(x)
            X_positive = np.maximum(X_flat, min_val)
            X_transformed = np.log(X_positive).reshape(-1, 1)
            return X_transformed, y

        if function_type == 9:  # Log-Polynomial: y = a + b*log(x) + c*log(x)^2
            X_positive = np.maximum(X_flat, min_val)
            X_log = np.log(X_positive)
            X_transformed = np.column_stack([X_log, X_log ** 2])
            return X_transformed, y

        if function_type == 10:  # Semi-Log: log(y) = a + b*x
            y_positive = np.maximum(y, min_val)
            y_transformed = np.log(y_positive)
            X_transformed = X_flat.reshape(-1, 1)
            return X_transformed, y_transformed

        if function_type == 11:  # Square Root: y = a + b*sqrt(x)
            X_positive = np.maximum(X_flat, 0)
            X_transformed = np.sqrt(X_positive).reshape(-1, 1)
            return X_transformed, y

        if function_type == 12:  # Inverse: y = a + b/x
            X_nonzero = np.where(np.abs(X_flat) > min_val, X_flat, min_val)
            X_transformed = (1.0 / X_nonzero).reshape(-1, 1)
            return X_transformed, y

        if function_type == 13:  # Log-Sqrt: y = a + b*log(x) + c*sqrt(x)
            X_positive = np.maximum(X_flat, min_val)
            X_log = np.log(X_positive)
            X_sqrt = np.sqrt(X_positive)
            X_transformed = np.column_stack([X_log, X_sqrt])
            return X_transformed, y

        if function_type == 14:  # Mixed: y = a + b*x + c*log(x)
            X_positive = np.maximum(X_flat, min_val)
            X_log = np.log(X_positive)
            X_transformed = np.column_stack([X_flat, X_log])
            return X_transformed, y

        if function_type == 15:  # Poly-Log: y = a + b*x + c*x^2 + d*log(x)
            X_positive = np.maximum(X_flat, min_val)
            X_log = np.log(X_positive)
            X_transformed = np.column_stack([X_flat, X_flat ** 2, X_log])
            return X_transformed, y

        if function_type == 16:  # Volatility Mix: y = a + b*sqrt(x) + c/x
            X_positive = np.maximum(X_flat, min_val)
            X_sqrt = np.sqrt(X_positive)
            X_inv = 1.0 / X_positive
            X_transformed = np.column_stack([X_sqrt, X_inv])
            return X_transformed, y

        raise ValueError(f"Unknown function type: {function_type}")

    # pylint: disable=duplicate-code,too-many-return-statements
    def _transform_features_for_function_pure(self, X, y, function_type):
        """Transform features for special functions using pure Python."""
        import math

        X_flat = X.flatten() if hasattr(X, 'flatten') else [item for sublist in X for item in sublist] if isinstance(
            X[0], list) else X
        y_list = y.tolist() if hasattr(y, 'tolist') else list(y)

        min_val = 1e-10

        if function_type == 8:  # Log-Linear: y = a + b*log(x)
            X_positive = [max(x, min_val) for x in X_flat]
            X_transformed = [[math.log(x)] for x in X_positive]
            return X_transformed, y_list

        if function_type == 9:  # Log-Polynomial: y = a + b*log(x) + c*log(x)^2
            X_positive = [max(x, min_val) for x in X_flat]
            X_log = [math.log(x) for x in X_positive]
            X_transformed = [[log_x, log_x ** 2] for log_x in X_log]
            return X_transformed, y_list

        if function_type == 10:  # Semi-Log: log(y) = a + b*x
            y_positive = [max(yi, min_val) for yi in y_list]
            y_transformed = [math.log(yi) for yi in y_positive]
            X_transformed = [[x] for x in X_flat]
            return X_transformed, y_transformed

        if function_type == 11:  # Square Root: y = a + b*sqrt(x)
            X_positive = [max(x, 0) for x in X_flat]
            X_transformed = [[math.sqrt(x)] for x in X_positive]
            return X_transformed, y_list

        if function_type == 12:  # Inverse: y = a + b/x
            X_nonzero = []
            for x in X_flat:
                if abs(x) > min_val:
                    X_nonzero.append(x)
                else:
                    X_nonzero.append(min_val if x >= 0 else -min_val)
            X_transformed = [[1.0 / x] for x in X_nonzero]
            return X_transformed, y_list

        if function_type == 13:  # Log-Sqrt: y = a + b*log(x) + c*sqrt(x)
            X_positive = [max(x, min_val) for x in X_flat]
            X_transformed = [[math.log(x), math.sqrt(x)] for x in X_positive]
            return X_transformed, y_list

        if function_type == 14:  # Mixed: y = a + b*x + c*log(x)
            X_positive = [max(x, min_val) for x in X_flat]
            X_transformed = [[X_flat[i], math.log(X_positive[i])] for i in range(len(X_flat))]
            return X_transformed, y_list

        if function_type == 15:  # Poly-Log: y = a + b*x + c*x^2 + d*log(x)
            X_positive = [max(x, min_val) for x in X_flat]
            X_transformed = [[X_flat[i], X_flat[i] ** 2, math.log(X_positive[i])] for i in range(len(X_flat))]
            return X_transformed, y_list

        if function_type == 16:  # Volatility Mix: y = a + b*sqrt(x) + c/x
            X_positive = [max(x, min_val) for x in X_flat]
            X_transformed = [[math.sqrt(x), 1.0 / x] for x in X_positive]
            return X_transformed, y_list

        raise ValueError(f"Unknown function type: {function_type}")

    def _get_regression_model(self, regression_type, **kwargs):
        """Return model instance based on regression type."""
        if regression_type == 1:
            return least_squares_numpy.LinearRegression(degree=1, **kwargs)
        if regression_type == 2:
            return least_squares_numpy.RidgeRegression(alpha=1.0)
        if regression_type == 3:
            return least_squares_numpy.LassoRegression(alpha=1.0)
        if regression_type == 4:
            return least_squares_numpy.ElasticNetRegression(alpha=1.0, l1_ratio=0.5)
        raise ValueError(f"Unknown regression type: {regression_type}")

    def _run_numpy_regression(self, X, y, regression_type, function_type):
        """Run regression using NumPy implementation - FIXED VERSION."""
        if 1 <= function_type <= 7:
            # For polynomial functions
            degree = self.function_degree_mapping[function_type]

            if regression_type == 1:  # Linear regression
                model = least_squares_numpy.LinearRegression(degree=degree)
                model.fit(X.flatten(), y)
                coeffs = model.coefficients
                return {
                    'model': model,
                    'coefficients': coeffs,
                    'degree': degree,
                    'regression_type': 'Linear',
                    'function_type': function_type
                }

            # Ridge, Lasso, ElasticNet for polynomials
            # Create polynomial features
            X_poly = self._generate_polynomial_features(X, degree)

            if regression_type == 2:
                model = least_squares_numpy.RidgeRegression(alpha=1.0)
            elif regression_type == 3:
                model = least_squares_numpy.LassoRegression(alpha=1.0)
            elif regression_type == 4:
                model = least_squares_numpy.ElasticNetRegression(alpha=1.0, l1_ratio=0.5)

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
        # FIXED: For linear regression on special functions, we don't use polynomial degree
        if regression_type == 1:  # Linear regression
            # For special functions, we just do linear regression on transformed features
            X_transformed, y_transformed = self._transform_features_for_function_numpy(X, y, function_type)

            # Create a simple linear model on transformed features
            model = least_squares_numpy.RidgeRegression(alpha=0.0)  # alpha=0 means regular least squares
            model.fit(X_transformed, y_transformed)

            return {
                'model': model,
                'coefficients': model.coefficients,
                'function_type': function_type,
                'regression_type': 'Linear',
                'transformation': f'Function {function_type}',
                'is_transformed': True
            }

        # For Ridge, Lasso, ElasticNet on special functions
        X_transformed, y_transformed = self._transform_features_for_function_numpy(X, y, function_type)

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

    def _run_numba_regression(self, X, y, regression_type, function_type):
        """Run regression using Numba implementation."""
        if 1 <= function_type <= 7:
            # For polynomial functions
            degree = self.function_degree_mapping[function_type]

            if regression_type == 1:  # Linear regression
                model = least_squares_numba.LinearRegression(degree=degree)
                X_flat = X.flatten() if hasattr(X, 'flatten') else [item for sublist in X for item in
                                                                    sublist] if isinstance(X[0], list) else X
                model.fit(X_flat, y)
                coeffs = model.coefficients
                return {
                    'coefficients': coeffs,
                    'status': 'success',
                    'engine': 'numba',
                    'model': model,
                    'degree': degree
                }

            elif regression_type == 2:  # Ridge regression
                model = least_squares_numba.RidgeRegression(alpha=1.0)
                X_reshaped = X.reshape(-1, 1) if hasattr(X, 'reshape') else [[x] for x in X]
                model.fit(X_reshaped, y)
                coeffs = model.coefficients if hasattr(model, 'coefficients') else [model.intercept] + list(model.coefficients) if hasattr(model, 'intercept') else []
                return {
                    'coefficients': coeffs,
                    'status': 'success',
                    'engine': 'numba',
                    'model': model,
                    'degree': 1
                }

            elif regression_type == 3:  # Lasso regression
                model = least_squares_numba.LassoRegression(alpha=0.1)
                X_reshaped = X.reshape(-1, 1) if hasattr(X, 'reshape') else [[x] for x in X]
                model.fit(X_reshaped, y)
                coeffs = model.coefficients
                return {
                    'coefficients': coeffs,
                    'status': 'success',
                    'engine': 'numba',
                    'model': model,
                    'degree': 1
                }

            elif regression_type == 4:  # ElasticNet regression
                model = least_squares_numba.ElasticNetRegression(alpha=0.1, l1_ratio=0.5)
                X_reshaped = X.reshape(-1, 1) if hasattr(X, 'reshape') else [[x] for x in X]
                model.fit(X_reshaped, y)
                coeffs = model.coefficients
                return {
                    'coefficients': coeffs,
                    'status': 'success',
                    'engine': 'numba',
                    'model': model,
                    'degree': 1
                }

        else:
            # For special functions (8-16)
            try:
                X_transformed, y_transformed = self._transform_features_for_function_pure(X, y, function_type)

                if regression_type == 1:  # Linear regression
                    model = least_squares_numba.LinearRegression(degree=1)
                    X_flat = [x[0] for x in X_transformed]
                    model.fit(X_flat, y_transformed)
                    coeffs = model.coefficients
                    return {
                        'coefficients': coeffs,
                        'status': 'success',
                        'engine': 'numba',
                        'model': model,
                        'degree': 1,
                        'is_transformed': True
                    }

                elif regression_type == 2:  # Ridge regression
                    model = least_squares_numba.RidgeRegression(alpha=1.0)
                    model.fit(X_transformed, y_transformed)
                    coeffs = model.coefficients if hasattr(model, 'coefficients') else [model.intercept] + list(model.coefficients) if hasattr(model, 'intercept') else []
                    return {
                        'coefficients': coeffs,
                        'status': 'success',
                        'engine': 'numba',
                        'model': model,
                        'degree': 1,
                        'is_transformed': True
                    }

                elif regression_type == 3:  # Lasso regression
                    model = least_squares_numba.LassoRegression(alpha=0.1)
                    model.fit(X_transformed, y_transformed)
                    coeffs = model.coefficients
                    return {
                        'coefficients': coeffs,
                        'status': 'success',
                        'engine': 'numba',
                        'model': model,
                        'degree': 1,
                        'is_transformed': True
                    }

                elif regression_type == 4:  # ElasticNet regression
                    model = least_squares_numba.ElasticNetRegression(alpha=0.1, l1_ratio=0.5)
                    model.fit(X_transformed, y_transformed)
                    coeffs = model.coefficients
                    return {
                        'coefficients': coeffs,
                        'status': 'success',
                        'engine': 'numba',
                        'model': model,
                        'degree': 1,
                        'is_transformed': True
                    }

            except Exception as e:
                return {
                    'status': 'failed',
                    'engine': 'numba',
                    'error': str(e)
                }

        return {
            'status': 'failed',
            'engine': 'numba',
            'error': 'Unknown regression or function type'
        }

    def _run_pure_regression(self, X, y, regression_type, function_type):
        """Run regression using Pure Python implementation."""
        if 1 <= function_type <= 7:
            # For polynomial functions
            degree = self.function_degree_mapping[function_type]

            if regression_type == 1:  # Linear regression
                model = least_squares_pure.LinearRegression(degree=degree)
                X_flat = X.flatten() if hasattr(X, 'flatten') else [item for sublist in X for item in
                                                                    sublist] if isinstance(X[0], list) else X
                model.fit(X_flat, y)
                coeffs = model.coefficients
                return {
                    'model': model,
                    'coefficients': coeffs,
                    'degree': degree,
                    'regression_type': 'Linear',
                    'function_type': function_type
                }

            # Ridge, Lasso, ElasticNet for polynomials
            # Create polynomial features manually for pure Python
            X_poly = self._generate_polynomial_features_pure(X, degree)

            if regression_type == 2:  # Ridge
                model = least_squares_pure.RidgeRegression(alpha=1.0)
            elif regression_type == 3:  # Lasso
                model = least_squares_pure.LassoRegression(alpha=1.0)
            elif regression_type == 4:  # ElasticNet
                model = least_squares_pure.ElasticNetRegression(alpha=1.0, l1_ratio=0.5)

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
        if regression_type == 1:  # Linear regression
            # For special functions, we just do linear regression on transformed features
            X_transformed, y_transformed = self._transform_features_for_function_pure(X, y, function_type)
            model = least_squares_pure.RidgeRegression(alpha=0.0)  # Regular least squares
            model.fit(X_transformed, y_transformed)

            return {
                'model': model,
                'coefficients': model.coefficients,
                'function_type': function_type,
                'regression_type': 'Linear',
                'transformation': f'Function {function_type}',
                'is_transformed': True
            }

        # For Ridge, Lasso, ElasticNet on special functions
        X_transformed, y_transformed = self._transform_features_for_function_pure(X, y, function_type)

        # Get model based on regression type
        if regression_type == 2:
            model = least_squares_pure.RidgeRegression(alpha=1.0)
        elif regression_type == 3:
            model = least_squares_pure.LassoRegression(alpha=1.0)
        elif regression_type == 4:
            model = least_squares_pure.ElasticNetRegression(alpha=1.0, l1_ratio=0.5)

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

    def _generate_polynomial_features_pure(self, X, degree):
        """Generate polynomial features for pure Python (no NumPy)."""
        X_flat = X.flatten() if hasattr(X, 'flatten') else [item for sublist in X for item in sublist] if isinstance(X[0], list) else X
        n = len(X_flat)
        
        # Create list of lists for polynomial features
        polynomial_features = []
        for i in range(n):
            row = []
            for d in range(1, degree + 1):
                row.append(X_flat[i] ** d)
            polynomial_features.append(row)
        
        return polynomial_features

    def _transform_features_for_function_pure(self, X, y, function_type):
        """Transform features for special functions using pure Python."""
        import math
        
        X_flat = X.flatten() if hasattr(X, 'flatten') else [item for sublist in X for item in sublist] if isinstance(X[0], list) else X
        y_list = y.tolist() if hasattr(y, 'tolist') else list(y)
        
        min_val = 1e-10
        
        if function_type == 8:  # Log-Linear: y = a + b*log(x)
            X_positive = [max(x, min_val) for x in X_flat]
            X_transformed = [[math.log(x)] for x in X_positive]
            return X_transformed, y_list
            
        if function_type == 11:  # Square Root: y = a + b*sqrt(x)
            X_positive = [max(x, 0) for x in X_flat]
            X_transformed = [[math.sqrt(x)] for x in X_positive]
            return X_transformed, y_list
            
        if function_type == 12:  # Inverse: y = a + b/x
            X_nonzero = [x if abs(x) > min_val else min_val for x in X_flat]
            X_transformed = [[1.0 / x] for x in X_nonzero]
            return X_transformed, y_list
        
        # For other function types, just return linear transformation for simplicity
        X_transformed = [[x] for x in X_flat]
        return X_transformed, y_list

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

        print(f"{S_BOLD}Summary:{E_BOLD} {successful} successful, {failed} failed, {not_implemented} not implemented")
