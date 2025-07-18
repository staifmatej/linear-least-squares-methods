"""Unified regression engine for multiple implementations and regression types."""

import math
import numpy as np

from approaches import least_squares_numpy
from approaches import least_squares_numba
from approaches import least_squares_pure
from constants import S_BOLD, E_BOLD


class RegressionRun:
    """Unified regression engine that supports multiple implementations and regression types."""

    def __init__(self, engine_choice, regression_types, function_types):
        self.engine_choice = engine_choice
        self.regression_types = regression_types
        self.function_types = function_types
        self.results = {}

        self.engine_mapping = {
            1: "numpy",
            2: "numba",
            3: "pure"
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
        if self.engine_choice == 1:
            return self._run_numpy_regression(X, y, regression_type, function_type)
        if self.engine_choice == 2:
            return self._run_numba_regression(X, y, regression_type, function_type)
        if self.engine_choice == 3:
            return self._run_pure_regression(X, y, regression_type, function_type)
        raise ValueError(f"Unknown engine choice: {self.engine_choice}")

    # pylint: disable=too-many-return-statements
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
            return least_squares_numpy.RidgeRegression(alpha=0.001)
        if regression_type == 3:
            return least_squares_numpy.LassoRegression(alpha=0.0001)
        if regression_type == 4:
            return least_squares_numpy.ElasticNetRegression(alpha=0.0001, l1_ratio=0.5)
        raise ValueError(f"Unknown regression type: {regression_type}")

    # pylint: disable=too-many-return-statements
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
                    'function_type': function_type,
                    'condition_number': getattr(model, 'condition_number', None)
                }

            # Ridge, Lasso, ElasticNet for polynomials
            # Create polynomial features
            X_poly = self._generate_polynomial_features(X, degree)

            if regression_type == 2:
                # Pro vysoké stupně použít menší alpha
                alpha = 0.001 if degree <= 5 else 0.00001
                model = least_squares_numpy.RidgeRegression(alpha=alpha)
                # Store degree for polynomial prediction
                model.degree = degree
            elif regression_type == 3:
                alpha = 0.0001 if degree <= 5 else 0.000001
                model = least_squares_numpy.LassoRegression(alpha=alpha)
                # Store degree for polynomial prediction
                model.degree = degree
            elif regression_type == 4:
                alpha = 0.0001 if degree <= 5 else 0.000001
                model = least_squares_numpy.ElasticNetRegression(alpha=alpha, l1_ratio=0.5)
                # Store degree for polynomial prediction
                model.degree = degree

            model.fit(X_poly, y)
            coeffs = model.coefficients

            return {
                'model': model,
                'coefficients': coeffs,
                'degree': degree,
                'regression_type': self.regression_mapping[regression_type],
                'function_type': function_type,
                'condition_number': getattr(model, 'condition_number', None)
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
                'is_transformed': True,
                'condition_number': getattr(model, 'condition_number', None)
            }

        # For Ridge, Lasso, ElasticNet on special functions
        X_transformed, y_transformed = self._transform_features_for_function_numpy(X, y, function_type)

        # Get model based on regression type
        if regression_type == 2:
            model = least_squares_numpy.RidgeRegression(alpha=0.1)
        elif regression_type == 3:
            model = least_squares_numpy.LassoRegression(alpha=0.01)
        elif regression_type == 4:
            model = least_squares_numpy.ElasticNetRegression(alpha=0.01, l1_ratio=0.5)
        else:
            raise ValueError(f"Unknown regression type: {regression_type}")

        # Fit model
        model.fit(X_transformed, y_transformed)

        return {
            'model': model,
            'coefficients': model.coefficients,
            'function_type': function_type,
            'regression_type': self.regression_mapping[regression_type],
            'transformation': f'Function {function_type}',
            'is_transformed': True,
            'condition_number': getattr(model, 'condition_number', None)
        }

    def _generate_polynomial_features(self, X, degree):
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
            # ODSTRANIT nebo ZMIRNIT škálování - žádné dodatečné škálování
            feature = X_normalized ** d  # Žádné dodatečné škálování
            polynomial_features.append(feature)

        return np.column_stack(polynomial_features)

    # pylint: disable=too-many-return-statements
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
                    'degree': degree,
                    'condition_number': getattr(model, 'condition_number', None)
                }

            if regression_type == 2:  # Ridge regression
                # Pro vysoké stupně použít menší alpha
                alpha = 0.001 if degree <= 5 else 0.00001
                model = least_squares_numba.RidgeRegression(alpha=alpha, degree=degree)

                # Create polynomial features (consistent with NumPy)
                X_poly_numpy = self._generate_polynomial_features(X, degree)
                # Convert NumPy array to list of lists for Numba engine
                X_poly = X_poly_numpy.tolist()
                model.fit(X_poly, y)
                coeffs = model.coefficients
                return {
                    'coefficients': coeffs,
                    'status': 'success',
                    'engine': 'numba',
                    'model': model,
                    'degree': degree,
                    'condition_number': getattr(model, 'condition_number', None)
                }

            if regression_type == 3:  # Lasso regression
                alpha = 0.0001 if degree <= 5 else 0.000001
                model = least_squares_numba.LassoRegression(alpha=alpha)

                # Create polynomial features (consistent with NumPy)
                X_poly_numpy = self._generate_polynomial_features(X, degree)
                # Convert NumPy array to list of lists for Numba engine
                X_poly = X_poly_numpy.tolist()
                model.fit(X_poly, y)
                coeffs = model.coefficients
                return {
                    'coefficients': coeffs,
                    'status': 'success',
                    'engine': 'numba',
                    'model': model,
                    'degree': degree,
                    'condition_number': getattr(model, 'condition_number', None)
                }

            if regression_type == 4:  # ElasticNet regression
                alpha = 0.0001 if degree <= 5 else 0.000001
                model = least_squares_numba.ElasticNetRegression(alpha=alpha, l1_ratio=0.5, degree=degree)
                X_flat = X.flatten() if hasattr(X, 'flatten') else [item for sublist in X for item in
                                                                    sublist] if isinstance(X[0], list) else X
                model.fit(X_flat, y)
                coeffs = model.coefficients
                return {
                    'coefficients': coeffs,
                    'status': 'success',
                    'engine': 'numba',
                    'model': model,
                    'degree': degree,
                    'condition_number': getattr(model, 'condition_number', None)
                }

        else:
            # For special functions (8-16)
            try:
                X_transformed, y_transformed = self._transform_features_for_function_pure(X, y, function_type)

                if regression_type == 1:  # Linear regression
                    model = least_squares_numba.LinearRegression(degree=1)
                    model.fit(X_transformed, y_transformed)
                    coeffs = model.coefficients
                    return {
                        'coefficients': coeffs,
                        'status': 'success',
                        'engine': 'numba',
                        'model': model,
                        'degree': 1,
                        'is_transformed': True,
                        'condition_number': getattr(model, 'condition_number', None)
                    }

                if regression_type == 2:  # Ridge regression
                    model = least_squares_numba.RidgeRegression(alpha=0.001)
                    model.fit(X_transformed, y_transformed)
                    coeffs = model.coefficients if hasattr(model, 'coefficients') else [model.intercept] + list(model.coefficients) if hasattr(model, 'intercept') else []
                    return {
                        'coefficients': coeffs,
                        'status': 'success',
                        'engine': 'numba',
                        'model': model,
                        'degree': 1,
                        'is_transformed': True,
                        'condition_number': getattr(model, 'condition_number', None)
                    }

                if regression_type == 3:  # Lasso regression
                    model = least_squares_numba.LassoRegression(alpha=0.1)
                    model.fit(X_transformed, y_transformed)
                    coeffs = model.coefficients
                    return {
                        'coefficients': coeffs,
                        'status': 'success',
                        'engine': 'numba',
                        'model': model,
                        'degree': 1,
                        'is_transformed': True,
                        'condition_number': getattr(model, 'condition_number', None)
                    }

                if regression_type == 4:  # ElasticNet regression
                    model = least_squares_numba.ElasticNetRegression(alpha=0.1, l1_ratio=0.5)
                    model.fit(X_transformed, y_transformed)
                    coeffs = model.coefficients
                    return {
                        'coefficients': coeffs,
                        'status': 'success',
                        'engine': 'numba',
                        'model': model,
                        'degree': 1,
                        'is_transformed': True,
                        'condition_number': getattr(model, 'condition_number', None)
                    }

            except (ValueError, RuntimeError, ArithmeticError) as e:
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

    # pylint: disable=too-many-return-statements
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
            # Create polynomial features (consistent with NumPy)
            X_poly_numpy = self._generate_polynomial_features(X, degree)
            # Convert NumPy array to list of lists for Pure engine
            X_poly = X_poly_numpy.tolist()

            if regression_type == 2:  # Ridge
                # Pro vysoké stupně použít menší alpha
                alpha = 0.001 if degree <= 5 else 0.00001
                model = least_squares_pure.RidgeRegression(alpha=alpha)
            elif regression_type == 3:  # Lasso
                alpha = 0.0001 if degree <= 5 else 0.000001
                model = least_squares_pure.LassoRegression(alpha=alpha)
            elif regression_type == 4:  # ElasticNet
                alpha = 0.0001 if degree <= 5 else 0.000001
                model = least_squares_pure.ElasticNetRegression(alpha=alpha, l1_ratio=0.5)

            model.fit(X_poly, y)
            coeffs = model.coefficients

            return {
                'model': model,
                'coefficients': coeffs,
                'degree': degree,
                'regression_type': self.regression_mapping[regression_type],
                'function_type': function_type,
                'condition_number': getattr(model, 'condition_number', None)
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
                'is_transformed': True,
                'condition_number': getattr(model, 'condition_number', None)
            }

        # For Ridge, Lasso, ElasticNet on special functions
        X_transformed, y_transformed = self._transform_features_for_function_pure(X, y, function_type)

        # Get model based on regression type (special functions use degree=1)
        if regression_type == 2:
            model = least_squares_pure.RidgeRegression(alpha=0.001)
        elif regression_type == 3:
            model = least_squares_pure.LassoRegression(alpha=0.0001)
        elif regression_type == 4:
            model = least_squares_pure.ElasticNetRegression(alpha=0.0001, l1_ratio=0.5)
        else:
            raise ValueError(f"Unknown regression type: {regression_type}")

        # Fit model
        model.fit(X_transformed, y_transformed)

        return {
            'model': model,
            'coefficients': model.coefficients,
            'function_type': function_type,
            'regression_type': self.regression_mapping[regression_type],
            'transformation': f'Function {function_type}',
            'is_transformed': True,
            'condition_number': getattr(model, 'condition_number', None)
        }

    def _generate_polynomial_features_pure(self, X, degree):
        """Generate polynomial features for pure Python with stable scaling."""
        X_flat = X.flatten() if hasattr(X, 'flatten') else [item for sublist in X for item in sublist] if isinstance(X[0], list) else X
        n = len(X_flat)

        # Normalize to [0, 1] range for better numerical stability
        x_min = min(X_flat)
        x_max = max(X_flat)
        if x_max - x_min > 1e-10:
            X_normalized = [(x - x_min) / (x_max - x_min) for x in X_flat]
        else:
            X_normalized = X_flat

        # Create list of lists for polynomial features bez škálování
        polynomial_features = []
        for i in range(n):
            row = []
            for d in range(1, degree + 1):
                # ODSTRANIT nebo ZMIRNIT škálování - žádné dodatečné škálování
                feature = X_normalized[i] ** d  # Žádné dodatečné škálování
                row.append(feature)
            polynomial_features.append(row)

        return polynomial_features

    def _generate_polynomial_features_numba(self, X, degree):
        """Generate polynomial features for Numba engine (consistent with NumPy)."""
        X_flat = X.flatten() if hasattr(X, 'flatten') else [item for sublist in X for item in sublist] if isinstance(X[0], list) else X

        # Use same normalization as NumPy for consistency
        x_min = min(X_flat)
        x_max = max(X_flat)

        # Apply normalization for degree > 3 (consistent with NumPy)
        if degree > 3:
            if x_max - x_min > 1e-10:
                X_normalized = [2 * (x - x_min) / (x_max - x_min) - 1 for x in X_flat]
            else:
                X_normalized = X_flat
        else:
            X_normalized = X_flat

        # Create list of lists for polynomial features
        polynomial_features = []
        for i in range(len(X_flat)):
            row = []
            for d in range(1, degree + 1):
                feature = X_normalized[i] ** d
                row.append(feature)
            polynomial_features.append(row)

        return polynomial_features


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
