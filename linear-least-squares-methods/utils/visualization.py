"""Visualization and plotting utilities for regression results."""

import matplotlib.pyplot as plt
import numpy as np

# Global constants used for bold text and red warning messages.
S_BOLD = "\033[1m"
E_BOLD = "\033[0m"
S_RED = "\033[91m"
E_RED = "\033[0m"


class VisualizationData:
    """Class for visualizing regression results."""

    def __init__(self, X, y, results):
        self.X = X
        self.y = y
        self.results = results

    # pylint: disable=too-many-locals,too-many-branches
    def plot_results(self, selected_results=None):
        """Plot regression results."""
        if selected_results is None:
            selected_results = self.results

        # Filter successful results
        successful_results = [(k, v) for k, v in selected_results.items()
                              if v is not None and v.get('status') != 'not_implemented']

        if not successful_results:
            print("No successful results to plot!")
            return

        # Create subplot grid
        n_plots = len(successful_results)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        _fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Create x points for smooth curves
        x_smooth = np.linspace(self.X.min(), self.X.max(), 300)

        for idx, ((reg_type, func_type), result) in enumerate(successful_results):
            ax = axes[idx] if n_plots > 1 else axes[0]

            # Plot original data
            ax.scatter(self.X, self.y, alpha=0.5, label='Data')

            # Get predictions
            model = result['model']

            try:
                # FIXED: Handle predictions correctly for each function type
                if func_type >= 8:  # Special functions
                    if result.get('is_transformed', False):
                        # Transform x_smooth same as during fit
                        X_smooth_transformed, _ = self._transform_features_for_prediction(
                            x_smooth.reshape(-1, 1), func_type
                        )
                        y_pred_smooth = model.predict(X_smooth_transformed)

                        # For semi-log transformation, we need to exponentiate the result
                        if func_type == 10:
                            y_pred_smooth = np.exp(y_pred_smooth)
                    else:
                        y_pred_smooth = model.predict(x_smooth)
                else:
                    # For polynomials
                    if hasattr(model, 'predict') and model.__class__.__name__ == 'PolynomialRegression':
                        y_pred_smooth = model.predict(x_smooth)
                    else:
                        # For Ridge/Lasso/ElasticNet on polynomials
                        degree = result.get('degree', 1)
                        X_smooth_poly = self._generate_polynomial_features_for_plot(
                            x_smooth.reshape(-1, 1), degree, self.X
                        )
                        y_pred_smooth = model.predict(X_smooth_poly)

                ax.plot(x_smooth, y_pred_smooth, 'r-', label='Fit', linewidth=2)

            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                print(f"Error plotting {reg_type}, {func_type}: {e}")
                continue

            # Set up plot
            regression_names = {1: "Polynomial", 2: "Ridge", 3: "Lasso", 4: "ElasticNet"}
            function_names = {
                1: "Linear", 2: "Quadratic", 3: "Cubic", 4: "Quartic",
                5: "Quintic", 6: "Sextic", 7: "Septic",
                8: "Log-Linear", 9: "Log-Polynomial", 10: "Semi-Log",
                11: "Square Root", 12: "Inverse", 13: "Log-Sqrt",
                14: "Mixed", 15: "Poly-Log", 16: "Volatility Mix"
            }

            title = f"{regression_names[reg_type]} - {function_names.get(func_type, f'Function {func_type}')}"
            ax.set_title(title)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(len(successful_results), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

    def _generate_polynomial_features_for_plot(self, X, degree, X_original):
        """Generate polynomial features for plotting with same normalization as training."""
        X_flat = X.flatten() if X.ndim > 1 else X
        X_orig_flat = X_original.flatten()

        # Use same normalization as in training
        if degree > 3:
            x_min, x_max = X_orig_flat.min(), X_orig_flat.max()
            if x_max - x_min > 1e-10:
                X_normalized = 2 * (X_flat - x_min) / (x_max - x_min) - 1
            else:
                X_normalized = X_flat
        else:
            X_normalized = X_flat

        polynomial_features = []
        for d in range(1, degree + 1):
            if d > 5:
                feature = X_normalized ** d / (10 ** (d - 5))
            else:
                feature = X_normalized ** d
            polynomial_features.append(feature)

        return np.column_stack(polynomial_features)

    # pylint: disable=too-many-return-statements
    def _transform_features_for_prediction(self, X, function_type):
        """Transform features for prediction according to function type."""
        X = X.flatten()
        min_val = 1e-10

        if function_type == 8:  # Log-Linear
            X_positive = np.where(X > 0, X, min_val)
            X_transformed = np.log(X_positive).reshape(-1, 1)
            return X_transformed, None

        if function_type == 9:  # Log-Polynomial
            X_positive = np.where(X > 0, X, min_val)
            X_log = np.log(X_positive)
            X_transformed = np.column_stack([X_log, X_log ** 2])
            return X_transformed, None

        if function_type == 10:  # Semi-Log
            return X.reshape(-1, 1), None

        if function_type == 11:  # Square Root
            X_positive = np.where(X >= 0, X, 0)
            X_transformed = np.sqrt(X_positive).reshape(-1, 1)
            return X_transformed, None

        if function_type == 12:  # Inverse
            X_nonzero = np.where(np.abs(X) > min_val, X, min_val * np.sign(X))
            X_nonzero = np.where(X_nonzero == 0, min_val, X_nonzero)
            X_transformed = (1.0 / X_nonzero).reshape(-1, 1)
            return X_transformed, None

        if function_type == 13:  # Log-Sqrt
            X_positive = np.where(X > 0, X, min_val)
            X_log = np.log(X_positive)
            X_sqrt = np.sqrt(X_positive)
            X_transformed = np.column_stack([X_log, X_sqrt])
            return X_transformed, None

        if function_type == 14:  # Mixed
            X_positive = np.where(X > 0, X, min_val)
            X_log = np.log(X_positive)
            X_transformed = np.column_stack([X, X_log])
            return X_transformed, None

        if function_type == 15:  # Poly-Log
            X_positive = np.where(X > 0, X, min_val)
            X_log = np.log(X_positive)
            X_transformed = np.column_stack([X, X ** 2, X_log])
            return X_transformed, None

        if function_type == 16:  # Volatility Mix
            X_positive = np.where(X > 0, X, min_val)
            X_sqrt = np.sqrt(X_positive)
            X_inv = 1.0 / X_positive
            X_transformed = np.column_stack([X_sqrt, X_inv])
            return X_transformed, None

        raise ValueError(f"Unknown function type: {function_type}")

    def print_coefficients(self):
        """Print coefficients for all successful results."""
        print(f"\n{S_BOLD}=== Regression Coefficients ==={E_BOLD}")

        for (reg_type, func_type), result in self.results.items():
            if result is None or result.get('status') == 'not_implemented':
                continue

            coeffs = result.get('coefficients', [])
            regression_names = {1: "Polynomial", 2: "Ridge", 3: "Lasso", 4: "ElasticNet"}

            print(f"\n{regression_names[reg_type]} - Function {func_type}:")
            print(f"  Intercept: {coeffs[0]:.6f}")

            for i, coeff in enumerate(coeffs[1:], 1):
                print(f"  Coefficient {i}: {coeff:.6f}")
