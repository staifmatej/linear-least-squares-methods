"""Visualization and plotting utilities for regression results with Seaborn."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

# Suppress common warnings
warnings.filterwarnings('ignore', message='Objective did not converge')

# Set seaborn style for beautiful modern plots
sns.set_theme(style="whitegrid", palette="Set2")
sns.set_context("notebook", font_scale=1.2)

# Custom color palettes for different themes
COLOR_THEMES = {
    'modern': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
}

# Global constants used for bold text and red warning messages.
S_BOLD = "\033[1m"
E_BOLD = "\033[0m"
S_RED = "\033[91m"
E_RED = "\033[0m"


class VisualizationData:
    """Class for visualizing regression results with Seaborn."""

    def __init__(self, X, y, results):
        self.X = X
        self.y = y
        self.results = results

    def _ensure_numpy_array(self, data):
        """Convert data to numpy array if it's a list."""
        if isinstance(data, list):
            return np.array(data)
        return data

    def _transform_features_for_prediction_pure(self, X, function_type):
        """Transform features for prediction according to function type (pure Python version)."""
        import math

        X_list = X.flatten().tolist() if hasattr(X, 'flatten') else X
        min_val = 1e-10

        if function_type == 8:  # Log-Linear
            X_positive = [max(x, min_val) for x in X_list]
            X_transformed = [[math.log(x)] for x in X_positive]
            return X_transformed, None

        if function_type == 9:  # Log-Polynomial
            X_positive = [max(x, min_val) for x in X_list]
            X_log = [math.log(x) for x in X_positive]
            X_transformed = [[log_x, log_x ** 2] for log_x in X_log]
            return X_transformed, None

        if function_type == 10:  # Semi-Log
            X_transformed = [[x] for x in X_list]
            return X_transformed, None

        if function_type == 11:  # Square Root
            X_positive = [max(x, 0) for x in X_list]
            X_transformed = [[math.sqrt(x)] for x in X_positive]
            return X_transformed, None

        if function_type == 12:  # Inverse
            X_nonzero = []
            for x in X_list:
                if abs(x) > min_val:
                    X_nonzero.append(x)
                else:
                    X_nonzero.append(min_val if x >= 0 else -min_val)
            X_transformed = [[1.0 / x] for x in X_nonzero]
            return X_transformed, None

        if function_type == 13:  # Log-Sqrt
            X_positive = [max(x, min_val) for x in X_list]
            X_transformed = [[math.log(x), math.sqrt(x)] for x in X_positive]
            return X_transformed, None

        if function_type == 14:  # Mixed
            X_positive = [max(x, min_val) for x in X_list]
            X_transformed = [[X_list[i], math.log(X_positive[i])] for i in range(len(X_list))]
            return X_transformed, None

        if function_type == 15:  # Poly-Log
            X_positive = [max(x, min_val) for x in X_list]
            X_transformed = [[X_list[i], X_list[i] ** 2, math.log(X_positive[i])] for i in range(len(X_list))]
            return X_transformed, None

        if function_type == 16:  # Volatility Mix
            X_positive = [max(x, min_val) for x in X_list]
            X_transformed = [[math.sqrt(x), 1.0 / x] for x in X_positive]
            return X_transformed, None

        raise ValueError(f"Unknown function type: {function_type}")

    # pylint: disable=too-many-locals,too-many-branches, too-many-statements
    def plot_results(self, selected_results=None):
        """Plot regression results with enhanced Seaborn styling."""
        if selected_results is None:
            selected_results = self.results

        # Check if data is multidimensional
        if self.X.shape[1] > 1:
            print(f"{S_RED}Error{E_RED}: Visualization is not supported for multidimensional data!")
            print(f"Your data has {self.X.shape[1]} features. Visualization only works with 1 feature (2D plots).")
            return

        # Filter successful results
        successful_results = [(k, v) for k, v in selected_results.items()
                              if v is not None and v.get('status') != 'not_implemented']

        if not successful_results:
            print("No successful results to plot!")
            return

        # Create subplot grid with better spacing
        n_plots = len(successful_results)
        n_cols = min(3, n_plots)
        n_rows = max(1, (n_plots + n_cols - 1) // n_cols)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(7 * n_cols, 6 * n_rows), facecolor='white')

        theme_colors = COLOR_THEMES['modern']
        
        # Enhanced color scheme for different regression types
        scatter_color = '#3498db'  # Bright blue for data points - better visibility
        # Use same color for all regression types - consistent appearance
        consistent_color = theme_colors[0]  # Primary color for all
        line_colors = {
            "Linear": consistent_color,
            "Ridge": consistent_color,
            "Lasso": consistent_color,
            "ElasticNet": consistent_color
        }
        
        # Add gradient background
        fig.patch.set_facecolor('#fafafa')

        # Create x points for smooth curves
        x_smooth = np.linspace(self.X.min(), self.X.max(), 300)

        for idx, ((reg_type, func_type), result) in enumerate(successful_results):
            ax = plt.subplot(n_rows, n_cols, idx + 1)

            # Set beautiful individual subplot style
            ax.set_facecolor('#fcfcfc')
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.7, color='#e8e8e8')
            
            # Add subtle gradient border
            for spine in ax.spines.values():
                spine.set_edgecolor('#d0d0d0')
                spine.set_linewidth(1.5)

            # Plot original data with enhanced style - no overlapping effects
            scatter = ax.scatter(self.X.flatten(), self.y, alpha=0.9, s=60, 
                               c=scatter_color, edgecolors='white', linewidth=1.5,
                               label='Data Points', zorder=5)

            # Get predictions - handle different result formats
            if 'model' not in result:
                print(f"Warning: Result missing 'model' key: {result}")
                continue
            
            model = result['model']

            try:
                # Detect if model is from pure Python, numba, or sklearn wrapper implementation
                is_pure_python = hasattr(model, '__module__') and 'pure' in model.__module__
                is_numba_python = hasattr(model, '__module__') and 'numba' in model.__module__
                # Check if model uses sklearn internally
                is_sklearn_wrapper = (hasattr(model, 'model') and 
                                     hasattr(model.model, 'predict') and 
                                     model.__class__.__name__ in ['LassoRegression', 'ElasticNetRegression'])


                # Handle predictions correctly for each function type
                if func_type >= 8:  # Special functions
                    if result.get('is_transformed', False):
                        # Transform x_smooth same as during fit
                        if is_pure_python:
                            X_smooth_transformed, _ = self._transform_features_for_prediction_pure(
                                x_smooth.reshape(-1, 1), func_type
                            )
                        elif is_numba_python:
                            # For numba models, use pure Python transform (same approach)
                            X_smooth_transformed, _ = self._transform_features_for_prediction_pure(
                                x_smooth.reshape(-1, 1), func_type
                            )
                        elif is_sklearn_wrapper:
                            # For sklearn wrappers, use the underlying sklearn model directly
                            X_smooth_transformed, _ = self._transform_features_for_prediction(
                                x_smooth.reshape(-1, 1), func_type
                            )
                        else:
                            X_smooth_transformed, _ = self._transform_features_for_prediction(
                                x_smooth.reshape(-1, 1), func_type
                            )

                        if is_sklearn_wrapper and hasattr(model, 'model'):
                            # Use the underlying sklearn model for prediction
                            y_pred_smooth = model.model.predict(X_smooth_transformed)
                        elif is_numba_python or is_pure_python:
                            # For numba and pure python models, use direct prediction
                            y_pred_smooth = model.predict(X_smooth_transformed)
                        else:
                            y_pred_smooth = model.predict(X_smooth_transformed)

                        # Convert to numpy array if it's a list (from pure Python)
                        y_pred_smooth = self._ensure_numpy_array(y_pred_smooth)

                        # For semi-log transformation, we need to exponentiate the result
                        if func_type == 10:
                            y_pred_smooth = np.exp(y_pred_smooth)
                    else:
                        y_pred_smooth = model.predict(x_smooth)
                        y_pred_smooth = self._ensure_numpy_array(y_pred_smooth)
                else:
                    # For polynomials
                    if hasattr(model, 'predict') and model.__class__.__name__ == 'LinearRegression':
                        y_pred_smooth = model.predict(x_smooth)
                        y_pred_smooth = self._ensure_numpy_array(y_pred_smooth)
                    else:
                        # For Ridge/Lasso/ElasticNet on polynomials
                        degree = result.get('degree', 1)

                        if is_pure_python or is_numba_python:
                            # Use the same polynomial feature generation as the engine
                            from utils.run_regression import RegressionRun
                            runner = RegressionRun(1, [], [])  # Temporary runner to access method
                            X_smooth_poly_np = runner._generate_polynomial_features(x_smooth.reshape(-1, 1), degree)
                            X_smooth_poly = X_smooth_poly_np.tolist()

                            # For sklearn wrappers in Pure Python, use underlying sklearn model
                            if is_sklearn_wrapper and hasattr(model, 'model'):
                                # Convert to numpy for sklearn
                                X_smooth_poly_np = np.array(X_smooth_poly)
                                y_pred_smooth = model.model.predict(X_smooth_poly_np)
                            else:
                                y_pred_smooth = model.predict(X_smooth_poly)
                            y_pred_smooth = self._ensure_numpy_array(y_pred_smooth)
                        else:
                            # Use the same polynomial feature generation as the engine
                            from utils.run_regression import RegressionRun
                            runner = RegressionRun(1, [], [])  # Temporary runner to access method
                            X_smooth_poly = runner._generate_polynomial_features(x_smooth.reshape(-1, 1), degree)
                            
                            if is_sklearn_wrapper:
                                # Use the underlying sklearn model for prediction
                                y_pred_smooth = model.model.predict(X_smooth_poly)
                            else:
                                y_pred_smooth = model.predict(X_smooth_poly)
                            y_pred_smooth = self._ensure_numpy_array(y_pred_smooth)

                # Plot gorgeous fitted line with gradient effect
                regression_names = {1: "Linear", 2: "Ridge", 3: "Lasso", 4: "ElasticNet"}
                reg_name = regression_names[reg_type]
                line_color = line_colors.get(reg_name, theme_colors[0])

                # Main fit line with shadow effect
                ax.plot(x_smooth, y_pred_smooth, color='white', linewidth=5, 
                       alpha=0.8, zorder=2)  # White shadow
                ax.plot(x_smooth, y_pred_smooth, color=line_color, linewidth=3.5, 
                       alpha=0.95, label='Regression Fit', zorder=3)

                # Beautiful confidence band with gradient effect
                std_dev = np.std(y_pred_smooth)
                confidence_upper = y_pred_smooth + 0.08 * std_dev
                confidence_lower = y_pred_smooth - 0.08 * std_dev
                
                ax.fill_between(x_smooth, confidence_lower, confidence_upper,
                               alpha=0.15, color=line_color, zorder=1)
                
                # Add subtle highlight line
                ax.plot(x_smooth, y_pred_smooth, color='white', linewidth=1, 
                       alpha=0.6, zorder=4)

            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                print(f"Error plotting {reg_type}, {func_type}: {e}")
                continue

            # Set up plot with enhanced styling
            function_names = {
                1: "Linear", 2: "Quadratic", 3: "Cubic", 4: "Quartic",
                5: "Quintic", 6: "Sextic", 7: "Septic",
                8: "Log-Linear", 9: "Log-Polynomial", 10: "Semi-Log",
                11: "Square Root", 12: "Inverse", 13: "Log-Sqrt",
                14: "Mixed", 15: "Poly-Log", 16: "Volatility Mix"
            }

            title = f"{regression_names[reg_type]} • {function_names.get(func_type, f'Function {func_type}')}"

            # Beautiful modern title and labels with color coordination
            title_color = line_color if line_color else theme_colors[0]
            ax.set_title(title, fontsize=15, fontweight='bold', pad=25, 
                        color=title_color, alpha=0.9)
            ax.set_xlabel('Input Values (X)', fontsize=12, fontweight='semibold', 
                         color='#2c3e50', alpha=0.8)
            ax.set_ylabel('Output Values (Y)', fontsize=12, fontweight='semibold',
                         color='#2c3e50', alpha=0.8)

            # Gorgeous modern legend with transparency and style
            legend = ax.legend(loc='best', frameon=True, fancybox=True,
                              shadow=True, framealpha=0.95, fontsize=11,
                              edgecolor='none', facecolor='white')
            legend.get_frame().set_facecolor('#ffffff')
            legend.get_frame().set_alpha(0.95)
            
            # Add subtle border radius effect simulation
            legend.get_frame().set_linewidth(0)
            
            # Enhance tick parameters for modern look
            ax.tick_params(axis='both', which='major', labelsize=10, 
                          colors='#34495e', length=6, width=1)
            ax.tick_params(axis='both', which='minor', labelsize=8, 
                          colors='#7f8c8d', length=3, width=0.5)
            
            # Remove inner shadow - cleaner look

        # Hide empty subplots
        if n_plots < len(plt.gcf().axes):
            for idx in range(n_plots, len(plt.gcf().axes)):
                plt.gcf().axes[idx].set_visible(False)

        # Adjust layout and spacing for beautiful presentation
        plt.tight_layout(pad=4.0, h_pad=5.0, w_pad=5.0)

        # Remove main title for cleaner appearance

        # Add beautiful window styling
        plt.rcParams['figure.facecolor'] = '#fafafa'
        plt.rcParams['axes.facecolor'] = '#fcfcfc'
        
        # Show plot with enhanced modern window
        plt.show()

    def plot_results_individually(self, selected_results=None):
        """Plot each regression result in a separate window."""
        if selected_results is None:
            selected_results = self.results

        # Check if data is multidimensional
        if self.X.shape[1] > 1:
            print(f"{S_RED}Error{E_RED}: Visualization is not supported for multidimensional data!")
            print(f"Your data has {self.X.shape[1]} features. Visualization only works with 1 feature (2D plots).")
            return

        # Filter successful results
        successful_results = [(k, v) for k, v in selected_results.items()
                              if v is not None and v.get('status') != 'not_implemented']

        if not successful_results:
            print("No successful results to plot!")
            return

        print(f"Creating {len(successful_results)} individual plots...")

        # Choose color theme
        theme_colors = COLOR_THEMES['modern']
        scatter_color = '#3498db'
        consistent_color = theme_colors[0]

        # Create x points for smooth curves
        x_smooth = np.linspace(self.X.min(), self.X.max(), 300)

        for idx, ((reg_type, func_type), result) in enumerate(successful_results):
            # Create individual figure
            plt.figure(figsize=(8, 6), facecolor='#fafafa')
            ax = plt.gca()

            # Set beautiful subplot style
            ax.set_facecolor('#fcfcfc')
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.7, color='#e8e8e8')
            
            # Add subtle gradient border
            for spine in ax.spines.values():
                spine.set_edgecolor('#d0d0d0')
                spine.set_linewidth(1.5)

            # Plot original data
            scatter = ax.scatter(self.X.flatten(), self.y, alpha=0.9, s=80, 
                               c=scatter_color, edgecolors='white', linewidth=2,
                               label='Data Points', zorder=5)

            # Get predictions (same logic as main plot_results)
            model = result['model']

            try:
                # Detect model type
                is_pure_python = hasattr(model, '__module__') and 'pure' in model.__module__
                is_numba_python = hasattr(model, '__module__') and 'numba' in model.__module__
                is_sklearn_wrapper = (hasattr(model, 'model') and 
                                     hasattr(model.model, 'predict') and 
                                     model.__class__.__name__ in ['LassoRegression', 'ElasticNetRegression'])

                # Handle predictions (same logic as main method)
                if func_type >= 8:  # Special functions
                    if result.get('is_transformed', False):
                        if is_pure_python:
                            X_smooth_transformed, _ = self._transform_features_for_prediction_pure(
                                x_smooth.reshape(-1, 1), func_type
                            )
                        elif is_numba_python:
                            X_smooth_transformed, _ = self._transform_features_for_prediction_pure(
                                x_smooth.reshape(-1, 1), func_type
                            )
                        elif is_sklearn_wrapper:
                            X_smooth_transformed, _ = self._transform_features_for_prediction(
                                x_smooth.reshape(-1, 1), func_type
                            )
                        else:
                            X_smooth_transformed, _ = self._transform_features_for_prediction(
                                x_smooth.reshape(-1, 1), func_type
                            )

                        if is_sklearn_wrapper and hasattr(model, 'model'):
                            y_pred_smooth = model.model.predict(X_smooth_transformed)
                        elif is_numba_python or is_pure_python:
                            y_pred_smooth = model.predict(X_smooth_transformed)
                        else:
                            y_pred_smooth = model.predict(X_smooth_transformed)

                        y_pred_smooth = self._ensure_numpy_array(y_pred_smooth)

                        if func_type == 10:
                            y_pred_smooth = np.exp(y_pred_smooth)
                    else:
                        y_pred_smooth = model.predict(x_smooth)
                        y_pred_smooth = self._ensure_numpy_array(y_pred_smooth)
                else:
                    # For polynomials (same logic)
                    if hasattr(model, 'predict') and model.__class__.__name__ == 'LinearRegression':
                        y_pred_smooth = model.predict(x_smooth)
                        y_pred_smooth = self._ensure_numpy_array(y_pred_smooth)
                    else:
                        degree = result.get('degree', 1)
                        
                        if is_pure_python or is_numba_python:
                            x_smooth_list = x_smooth.tolist()
                            X_smooth_poly = []
                            for x in x_smooth_list:
                                row = []
                                for d in range(1, degree + 1):
                                    row.append(x ** d)
                                X_smooth_poly.append(row)
                            
                            y_pred_smooth = model.predict(X_smooth_poly)
                            y_pred_smooth = self._ensure_numpy_array(y_pred_smooth)
                        else:
                            # Use the same polynomial feature generation as the engine
                            from utils.run_regression import RegressionRun
                            runner = RegressionRun(1, [], [])  # Temporary runner to access method
                            X_smooth_poly = runner._generate_polynomial_features(x_smooth.reshape(-1, 1), degree)
                            
                            if is_sklearn_wrapper and hasattr(model, 'model'):
                                y_pred_smooth = model.model.predict(X_smooth_poly)
                            else:
                                y_pred_smooth = model.predict(X_smooth_poly)
                            y_pred_smooth = self._ensure_numpy_array(y_pred_smooth)

                # Plot gorgeous fitted line
                ax.plot(x_smooth, y_pred_smooth, color='white', linewidth=5, 
                       alpha=0.8, zorder=2)  # White shadow
                ax.plot(x_smooth, y_pred_smooth, color=consistent_color, linewidth=3.5, 
                       alpha=0.95, label='Regression Fit', zorder=3)

                # Beautiful confidence band
                std_dev = np.std(y_pred_smooth)
                confidence_upper = y_pred_smooth + 0.08 * std_dev
                confidence_lower = y_pred_smooth - 0.08 * std_dev
                
                ax.fill_between(x_smooth, confidence_lower, confidence_upper,
                               alpha=0.15, color=consistent_color, zorder=1)
                
                # Add subtle highlight line
                ax.plot(x_smooth, y_pred_smooth, color='white', linewidth=1, 
                       alpha=0.6, zorder=4)

            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                print(f"Error plotting {reg_type}, {func_type}: {e}")
                plt.close()
                continue

            # Set up plot titles and labels
            regression_names = {1: "Linear", 2: "Ridge", 3: "Lasso", 4: "ElasticNet"}
            function_names = {
                1: "Linear", 2: "Quadratic", 3: "Cubic", 4: "Quartic",
                5: "Quintic", 6: "Sextic", 7: "Septic",
                8: "Log-Linear", 9: "Log-Polynomial", 10: "Semi-Log",
                11: "Square Root", 12: "Inverse", 13: "Log-Sqrt",
                14: "Mixed", 15: "Poly-Log", 16: "Volatility Mix"
            }

            title = f"{regression_names[reg_type]} • {function_names.get(func_type, f'Function {func_type}')}"

            # Beautiful modern title and labels
            ax.set_title(title, fontsize=16, fontweight='bold', pad=25, 
                        color=consistent_color, alpha=0.9)
            ax.set_xlabel('Input Values (X)', fontsize=13, fontweight='semibold', 
                         color='#2c3e50', alpha=0.8)
            ax.set_ylabel('Output Values (Y)', fontsize=13, fontweight='semibold',
                         color='#2c3e50', alpha=0.8)

            # Gorgeous modern legend
            legend = ax.legend(loc='best', frameon=True, fancybox=True,
                              shadow=True, framealpha=0.95, fontsize=12,
                              edgecolor='none', facecolor='white')
            
            # Enhance tick parameters
            ax.tick_params(axis='both', which='major', labelsize=11, 
                          colors='#34495e', length=6, width=1)

            # Adjust layout
            plt.tight_layout(pad=3.0)

            # Show individual plot
            plt.show()

        print(f"Displayed {len(successful_results)} individual plots.")

    def _generate_polynomial_features_for_plot(self, X, degree, X_original):
        """Generate polynomial features for plotting with same normalization as training."""
        X_flat = X.flatten() if X.ndim > 1 else X
        X_orig_flat = X_original.flatten()

        # Use same normalization as in training - [0, 1] range
        x_min, x_max = X_orig_flat.min(), X_orig_flat.max()
        if x_max - x_min > 1e-10:
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

