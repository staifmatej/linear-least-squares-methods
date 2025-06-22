"""Main script for testing polynomial regression implementations."""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from approaches.least_squares_numpy import (
    PolynomialRegression, RidgeRegression,
    LassoRegression, ElasticNetRegression
)

# Global constants used for bold text and red warning messages.
S_BOLD = "\033[1m"
E_BOLD = "\033[0m"
S_RED = "\033[91m"
E_RED = "\033[0m"


def print_data_loaded(X,y):
    print("\n========= Data loaded =========")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("===============================\n")

def print_selected_specifications(engine_choice, regression_types, function_types):
    engine_mapping = {
        1: "approaches/least_squares_cpp.cpp",
        2: "approaches/least_squares_numpy.py",
        3: "approaches/least_squares_numba.py",
        4: "approaches/least_squares_pure.py"
    }

    print("\n==== Selected Configuration ===")
    print(f"Engine: {engine_mapping.get(engine_choice)}")
    print(f"Regression types: {regression_types}")
    print(f"Function types: {function_types}")
    print("===============================\n")

class DataLoader:
    """Class for loading data from various sources."""

    def __init__(self):
        pass

    def show_menu(self):
        """Show data input options menu."""
        print("\n===== Data Input Options =====")
        print("1. Manual input (2D points)")
        print("2. Manual input (multidimensional)")
        print("3. Load from CSV file")
        print("4. Generate synthetic data")
        print("5. Use example dataset")
        print("===============================")

        while 1:
            try:
                choice = int(input("\nChoose option (1-5): "))
                if 1 <= choice <= 5:
                    return choice
                print(f"{S_RED}Invalid input{E_RED}: Please enter number 1-5")
            except ValueError:
                print(f"{S_RED}Invalid input{E_RED}: Please enter a valid number")

    def input_2d_points(self):
        """Input 2D points manually."""

        print("\nEnter 2D points [x,y]. Type 'done' when finished.")
        print("Example: 1,2.5")

        x_data, y_data = [], []

        point_count = 0
        while 1:
            user_input = input("x,y: ").strip()
            if user_input.lower() == 'done':
                if point_count < 2:
                    print(f"\n{S_RED}Warning{E_RED}: Need at least 2 points!")
                    continue
                break

            try:
                x_2d_point, y_2d_point = map(float, user_input.split(','))
                x_data.append(x_2d_point)
                y_data.append(y_2d_point)
                print(f"Added point: [{x_2d_point}, {y_2d_point}]")
                point_count += 1
            except ValueError:
                print(f"{S_RED}Invalid input{E_RED}: Use x,y format (example: 1.5,2.3)")

        return np.array(x_data).reshape(-1, 1), np.array(y_data)

    def input_multidimensional(self):
        """Input multidimensional data manually."""
        print("\nMultidimensional data input")

        while 1:
            try:
                n_features = int(input("How many features (X dimensions): "))
                if n_features >= 1:
                    break
                print(f"{S_RED}Invalid input{E_RED}: Must be at least 1 feature")
            except ValueError:
                print(f"{S_RED}Invalid input{E_RED}: Please enter a valid number")

        while 1:
            print(f"\nEnter data points with {n_features} features + 1 target value")
            print(f"Format: x_1,x_2,...,x_{n_features},y")
            print("Type 'done' when finished")

            data_points = []

            while 1:
                user_input = input(f"x1,x2,...,x{n_features},y: ").strip()
                if user_input.lower() == 'done':
                    break

                try:
                    values = list(map(float, user_input.split(',')))
                    if len(values) != n_features + 1:
                        print(f"Expected {n_features + 1} values, got {len(values)}")
                        continue

                    data_points.append(values)
                    print(f"Added point: {values}")
                except ValueError:
                    print(f"{S_RED}Invalid format{E_RED}: Use numbers separated by commas")

            if len(data_points) >= 2:

                data_array = np.array(data_points)
                X_multidim = data_array[:, :-1]
                y_multidim = data_array[:, -1]
                return X_multidim, y_multidim
            else:
                print(f"{S_RED}Invalid input{E_RED}: Need at least 2 points! Please try again.")

    def _find_csv_files(self):
        """Find CSV files recursively in all subdirectories using os.walk()."""
        csv_files = []

        print(f"\n{S_BOLD}Searching for CSV files recursively...{E_BOLD}\n")
        for root, _, files in os.walk('.'):
            for file in files:
                if file.endswith('.csv'):
                    full_path = os.path.join(root, file)
                    clean_path = os.path.normpath(full_path)
                    csv_files.append(clean_path)
                    # print(f"  Found: {clean_path}")

        return csv_files

    def _csv_files_read_data(self, filename):
        """Read and process CSV file data with validation and column selection."""
        data = pd.read_csv(filename)

        # Check if file is empty
        if data.empty:
            print(f"\n{S_RED}Error{E_RED}: {filename} is empty!")
            raise ValueError("Empty file")

        # Check minimum requirements
        if data.shape[0] < 2:
            print(f"\n{S_RED}Error{E_RED}: {filename} has only {data.shape[0]} rows. Need at least 2 rows!")
            raise ValueError("Insufficient rows")

        if data.shape[1] < 2:
            print(f"\n{S_RED}Error{E_RED}: {filename} has only {data.shape[1]} columns. Need at least 2 columns!")
            raise ValueError("Insufficient columns")

        print(f"\nLoaded {filename} with shape: {data.shape}")
        print("First few rows:")
        print(data.head())

        print("\nColumn selection:")
        print("Columns:", list(data.columns))

        # Select target column
        y_col = input("Enter target column name (y): ").strip()
        if y_col not in data.columns:
            print(f"{S_RED}Warning{E_RED}: Column not found!")
            raise ValueError("Column not found")

        # Select feature columns
        print("Enter feature column names separated by comma")
        print("Or press Enter to use all except target column")
        x_cols_input = input("Feature columns: ").strip()

        if x_cols_input:
            x_cols = [col.strip() for col in x_cols_input.split(',')]
        else:
            x_cols = [col for col in data.columns if col != y_col]

        X_csv = data[x_cols].values
        y_csv = data[y_col].values

        print(f"\nSelected {len(x_cols)} features: {x_cols}")
        print(f"Target: {y_col}")

        return X_csv, y_csv

    def _csv_files_not_found(self, filename, csv_files):
        """Display CSV file selection menu and get user choice."""
        if len(csv_files) == 1:
            print("==== Found CSV file option ====")
        else:
            print("=== Found CSV files options ===")
        print(f"Found {len(csv_files)} CSV files:")
        for i, file in enumerate(csv_files, 1):
            print(f"{i}. {file}")
        print("===============================")

        try:
            choice = int(input("\nChoose file number or 0 for custom path: "))
            if choice == 0:
                filename = input("Enter full path to CSV file: ")
            else:
                filename = csv_files[choice - 1]
        except (ValueError, IndexError):
            filename = input("Enter full path to CSV file: ")
        return filename, csv_files

    def load_from_csv(self):
        """Load data from CSV file."""
        while 1:
            csv_files = self._find_csv_files()

            if not csv_files:
                print(f"\n{S_RED}Warning{E_RED}: No CSV files found in current directory or subdirectories!")
                filename = input("Enter full path to CSV file: ")
            else:
                filename = None
                filename, csv_files = self._csv_files_not_found(filename, csv_files)

            try:
                X_csv, y_csv = self._csv_files_read_data(filename)
                return X_csv, y_csv
            except FileNotFoundError:
                print(f"{S_RED}Warning{E_RED}: File {filename} not found! Please try again.")
            except (OSError, ValueError, KeyError) as e:
                print(f"{S_RED}Warning{E_RED}: Error loading file: {e}. Please try again.")

    def generate_synthetic_data(self):
        """Generate synthetic polynomial data."""
        print("\nSynthetic data generation")

        while 1:
            try:
                n_samples = int(input("Number of samples (default 50): ") or "50")
                if n_samples < 2:
                    print(f"{S_RED}Invalid input{E_RED}: Number of samples must be at least 2")
                    continue
                if n_samples > 1e10:
                    print(f"{S_RED}Invalid input{E_RED}: Number of samples too large (max. 1e10)")
                    continue
                break
            except ValueError:
                print(f"{S_RED}Invalid input{E_RED}: Please enter a valid number")

        while 1:
            try:
                poly_degree_synthetic = int(input("True polynomial degree (default 3): ") or "3")
                if poly_degree_synthetic < 1:
                    print(f"{S_RED}Invalid input{E_RED}: Polynomial degree must be at least 1")
                    continue
                if poly_degree_synthetic > 100:
                    print(f"{S_RED}Warning{E_RED}: Polynomial degree too extremely high (max. 100)")
                    continue
                break
            except ValueError:
                print(f"{S_RED}Invalid input{E_RED}: Please enter a valid number")

        while 1:
            try:
                noise_level = float(input("Noise level (default 0.1): ") or "0.1")
                if noise_level < 0:
                    print(f"{S_RED}Invalid input{E_RED}: Noise level cannot be negative")
                    continue
                if noise_level > 100:
                    print(f"{S_RED}Invalid input{E_RED}: Noise level too high (max. 100)")
                    continue
                break
            except ValueError:
                print(f"{S_RED}Invalid input{E_RED}: Please enter a valid number")

        np.random.seed(42)
        x_synthetic = np.linspace(-2, 2, n_samples)

        # Generate random coefficients
        true_coeffs = np.random.randn(poly_degree_synthetic + 1)

        # Generate polynomial
        y_synthetic_true = sum(coeff * x_synthetic ** i for i, coeff in enumerate(true_coeffs))
        y_synthetic_noisy = y_synthetic_true + noise_level * np.random.randn(n_samples)

        print(f"Generated {n_samples} points with degree {poly_degree_synthetic} polynomial")
        print(f"True coefficients: {true_coeffs}")

        return x_synthetic.reshape(-1, 1), y_synthetic_noisy

    def use_example_dataset(self):
        """Use predefined example dataset."""
        print("\n====== Example datasets =======")
        print("1. House prices (size vs price)")
        print("2. Temperature data (time vs temp)")
        print("3. Quadratic function")
        print("4. Sinusoidal function")
        print("===============================\n")

        while 1:
            try:
                choice = int(input("Choose example (1-4): "))
                if 1 <= choice <= 4:
                    break
                print(f"{S_RED}Invalid input{E_RED}: Please enter number 1-4")
            except ValueError:
                print(f"{S_RED}Invalid input{E_RED}: Please enter a valid number")

        if choice == 1:
            # House prices example
            x_house_size = np.array([50, 80, 100, 120, 150, 180, 200, 250, 300])
            y_house_price = np.array([150, 200, 240, 280, 320, 350, 380, 450, 500])
            return x_house_size.reshape(-1, 1), y_house_price

        elif choice == 2:
            # Temperature data
            x_time = np.linspace(0, 24, 25)  # 24 hours
            y_temperature = 15 + 10 * np.sin(2 * np.pi * x_time / 24) + np.random.normal(0, 1, 25)
            return x_time.reshape(-1, 1), y_temperature

        elif choice == 3:
            # Quadratic function
            x_quadratic = np.linspace(-3, 3, 20)
            y_quadratic = 2 * x_quadratic ** 2 + 3 * x_quadratic + 1 + np.random.normal(0, 2, 20)
            return x_quadratic.reshape(-1, 1), y_quadratic

        else:  # choice == 4
            # Sinusoidal function
            np.random.seed(42)
            x_sin = np.linspace(0, 4 * np.pi, 100)  # 100 points over 4π
            y_sin = 3 * np.sin(x_sin) + 0.5 * np.sin(3 * x_sin) + np.random.normal(0, 0.3, 100)
            return x_sin.reshape(-1, 1), y_sin

    def get_data(self):
        """Main method to get data based on user choice."""
        choice = self.show_menu()

        if choice == 1:
            return self.input_2d_points()
        if choice == 2:
            return self.input_multidimensional()
        if choice == 3:
            return self.load_from_csv()
        if choice == 4:
            return self.generate_synthetic_data()
        # default (choice == 5)
        return self.use_example_dataset()


class UserInputHandler:
    """Class for handling user input selection for regression engines, types, and functions."""

    def __init__(self):
        pass

    def get_engine_choice(self):
        """Get regression engine choice from user."""

        print("\n===== Regression Engines =====")
        print("1. C++ with ML Pack")
        print("2. Python with Numpy")
        print("3. Python with Numba")
        print("4. Pure Python with for-loops")
        print(f"{S_BOLD}* Lasso/ElasticNet use sklearn coordinate descent{E_BOLD}")
        print("===============================")

        while 1:
            try:
                choice = int(input("\nChoose option (1-4): "))
                if 1 <= choice <= 4:
                    return choice
                print(f"{S_RED}Invalid input{E_RED}: Please enter number 1-4")
            except ValueError:
                print(f"{S_RED}Invalid input{E_RED}: Please enter a valid number")

    def get_regression_types(self):
        """Get regression types selection from user."""

        print("\n===== Types of regression =====")
        print("1. Polynomial regression (Least Squares)")
        print("2. Ridge regression (Least Squares)")
        print("3. Lasso Regression (CoordinateDescent)")
        print("4. Elastic Net Regression (CoordinateDescent)")
        print("===============================\n")

        while 1:
            user_input = input(f"Enter types of regression you would you like to try: (for example: {S_BOLD}all{E_BOLD} or {S_BOLD}1,2,3{E_BOLD} or {S_BOLD}4{E_BOLD}): ").strip()
            
            if user_input.lower() == 'all':
                return [1, 2, 3, 4]
            
            try:
                choices = []
                for choice_str in user_input.split(','):
                    choice = int(choice_str.strip())
                    if 1 <= choice <= 4:
                        choices.append(choice)
                    else:
                        raise ValueError(f"{S_RED}Invalid input{E_RED}: Invalid choice: {choice}")
                
                if choices:
                    return list(set(choices))
                else:
                    raise ValueError(f"{S_RED}Invalid input{E_RED}: No valid choices!")
                    
            except ValueError:
                print(f"{S_RED}Invalid input{E_RED}: Please enter numbers 1-4 separated by commas or '{S_BOLD}all{E_BOLD}'")
                print(f"Example: {S_BOLD}1,2,3{E_BOLD} or {S_BOLD}all{E_BOLD}")

    def get_function_types(self):
        """Get function types selection from user."""

        print("\n==== Types of function fit ====")
        print("1. Linear (y = a + b*x)")                                 # Polynomial of degree 1.
        print("2. Quadratic (y = a + b*x + c*x^2)")                      # Polynomial of degree 2.
        print("3. Cubic (y = a + b*x + c*x^2 + d*x^3)")                  # Polynomial of degree 3.
        print("4. Quartic (y = a + b*x + c*x^2 + d*x^3 + e*x^4)")        # Polynomial of degree 4.
        print("5. Quintic (y = a + b*x + c*x^2 + ... + e*x^4 + f*x^5)")  # Polynomial of degree 5.
        print("6. Sextic (y = a + b*x + c*x^2 + ... + f*x^5 + g*x^6)")   # Polynomial of degree 6.
        print("7. Septic (y = a + b*x + ... + h*x^7)")                   # Polynomial of degree 7.
        print("8. Log-Linear (y = a + b*log(x))")                        # Interest rate curves.
        print("9. Log-Polynomial (y = a + b*log(x) + c*log(x)^2)")       # Volatility smile.
        print("10. Semi-Log (log(y) = a + b*x)")                          # Exponential growth.
        print("11. Square Root (y = a + b*sqrt(x))")                     # VIX, volatility √time.
        print("12. Inverse (y = a + b/x)")                               # Mean reversion speed.
        print("13. Log-Sqrt (y = a + b*log(x) + c*sqrt(x))")             # Complex volatility.
        print("14. Mixed (y = a + b*x + c*log(x))")                      # Yield curves.
        print("15. Poly-Log (y = a + b*x + c*x^2 + d*log(x))")           # Nelson-Siegel like.
        print("16. Volatility Mix (y = a + b*sqrt(x) + c/x)")            # GARCH approximation.
        print("===============================\n")

        while 1:
            user_input = input(f"Enter types of functions you would you like to fit: (for example: {S_BOLD}all{E_BOLD} or {S_BOLD}1,2,3{E_BOLD} or {S_BOLD}9{E_BOLD}): ").strip()
            
            if user_input.lower() == 'all':
                return list(range(1, 17))  # 1-16
            
            try:
                choices = []
                for choice_str in user_input.split(','):
                    choice = int(choice_str.strip())
                    if 1 <= choice <= 16:
                        choices.append(choice)
                    else:
                        raise ValueError(f"{S_RED}Invalid input{E_RED}: Invalid choice: {choice}")
                
                if choices:
                    return list(set(choices))
                else:
                    raise ValueError(f"{S_RED}Invalid input{E_RED}: No valid choices")
                    
            except ValueError:
                print(f"{S_RED}Invalid input{E_RED}: Please enter numbers 1-16 separated by commas or '{S_BOLD}all{E_BOLD}'")
                print(f"Example: {S_BOLD}1,2,3{E_BOLD} or {S_BOLD}all{E_BOLD}")


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
                    print(f"SUCCESS: {self.regression_mapping[reg_type]} + Function {func_type}")
                except Exception as e:
                    print(f"FAILED: {self.regression_mapping[reg_type]} + Function {func_type}: {str(e)}")
                    self.results[(reg_type, func_type)] = None

        return self.results

    def _run_single_regression(self, X, y, regression_type, function_type):
        """Run single regression based on engine choice."""
        if self.engine_choice == 2:
            return self._run_numpy_regression(X, y, regression_type, function_type)
        elif self.engine_choice == 1:
            return self._run_cpp_regression(X, y, regression_type, function_type)
        elif self.engine_choice == 3:
            return self._run_numba_regression(X, y, regression_type, function_type)
        elif self.engine_choice == 4:
            return self._run_pure_regression(X, y, regression_type, function_type)
        else:
            raise ValueError(f"Unknown engine choice: {self.engine_choice}")

    def _transform_features_for_function(self, X, y, function_type):
        """Transformuje features podle typu funkce."""
        X = X.flatten()

        # Ošetření negativních a nulových hodnot
        min_val = 1e-10

        if function_type == 8:  # Log-Linear: y = a + b*log(x)
            X_positive = np.where(X > 0, X, min_val)
            X_transformed = np.log(X_positive).reshape(-1, 1)
            return X_transformed, y

        elif function_type == 9:  # Log-Polynomial: y = a + b*log(x) + c*log(x)^2
            X_positive = np.where(X > 0, X, min_val)
            X_log = np.log(X_positive)
            X_transformed = np.column_stack([X_log, X_log ** 2])
            return X_transformed, y

        elif function_type == 10:  # Semi-Log: log(y) = a + b*x
            y_positive = np.where(y > 0, y, min_val)
            y_transformed = np.log(y_positive)
            return X.reshape(-1, 1), y_transformed

        elif function_type == 11:  # Square Root: y = a + b*sqrt(x)
            X_positive = np.where(X >= 0, X, 0)
            X_transformed = np.sqrt(X_positive).reshape(-1, 1)
            return X_transformed, y

        elif function_type == 12:  # Inverse: y = a + b/x
            X_nonzero = np.where(np.abs(X) > min_val, X, min_val * np.sign(X))
            X_nonzero = np.where(X_nonzero == 0, min_val, X_nonzero)
            X_transformed = (1.0 / X_nonzero).reshape(-1, 1)
            return X_transformed, y

        elif function_type == 13:  # Log-Sqrt: y = a + b*log(x) + c*sqrt(x)
            X_positive = np.where(X > 0, X, min_val)
            X_log = np.log(X_positive)
            X_sqrt = np.sqrt(X_positive)
            X_transformed = np.column_stack([X_log, X_sqrt])
            return X_transformed, y

        elif function_type == 14:  # Mixed: y = a + b*x + c*log(x)
            X_positive = np.where(X > 0, X, min_val)
            X_log = np.log(X_positive)
            X_transformed = np.column_stack([X, X_log])
            return X_transformed, y

        elif function_type == 15:  # Poly-Log: y = a + b*x + c*x^2 + d*log(x)
            X_positive = np.where(X > 0, X, min_val)
            X_log = np.log(X_positive)
            X_transformed = np.column_stack([X, X ** 2, X_log])
            return X_transformed, y

        elif function_type == 16:  # Volatility Mix: y = a + b*sqrt(x) + c/x
            X_positive = np.where(X > 0, X, min_val)
            X_sqrt = np.sqrt(X_positive)
            X_inv = 1.0 / X_positive
            X_transformed = np.column_stack([X_sqrt, X_inv])
            return X_transformed, y

        else:
            raise ValueError(f"Unknown function type: {function_type}")

    def _get_regression_model(self, regression_type, **kwargs):
        """Vrátí instanci modelu podle typu regrese."""
        if regression_type == 1:
            return PolynomialRegression(degree=1, **kwargs)
        elif regression_type == 2:
            return RidgeRegression(alpha=1.0)
        elif regression_type == 3:
            return LassoRegression(alpha=1.0)
        elif regression_type == 4:
            return ElasticNetRegression(alpha=1.0, l1_ratio=0.5)
        else:
            raise ValueError(f"Unknown regression type: {regression_type}")

    def _run_numpy_regression(self, X, y, regression_type, function_type):
        """Run regression using NumPy implementation."""
        if 1 <= function_type <= 7:
            # Pro polynomiální funkce
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

            else:  # Ridge, Lasso, ElasticNet pro polynomy
                # Vytvoříme polynomiální features
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

        else:
            # Pro speciální funkce 8-16
            X_transformed, y_transformed = self._transform_features_for_function(X, y, function_type)

            # Získáme model podle typu regrese
            model = self._get_regression_model(regression_type)

            # Fit model
            model.fit(X_transformed, y_transformed)

            return {
                'model': model,
                'coefficients': model.coefficients,
                'function_type': function_type,
                'regression_type': self.regression_mapping[regression_type],
                'transformation': f'Function {function_type}'
            }

    def _generate_polynomial_features(self, X, degree):
        """Generate polynomial features up to specified degree."""
        X_flat = X.flatten() if X.ndim > 1 else X

        # Normalizace pro vysoké stupně polynomů
        if degree > 3:
            x_min, x_max = X_flat.min(), X_flat.max()
            if x_max - x_min > 1e-10:
                X_normalized = 2 * (X_flat - x_min) / (x_max - x_min) - 1
            else:
                X_normalized = X_flat
        else:
            X_normalized = X_flat

        polynomial_features = []
        for d in range(1, degree + 1):
            polynomial_features.append(X_normalized ** d)

        return np.column_stack(polynomial_features)

    def _run_cpp_regression(self, X, y, regression_type, function_type):
        """Run regression using C++ implementation."""
        print(f"  C++ implementation not yet available")
        return {
            'status': 'not_implemented',
            'engine': 'cpp',
            'message': 'C++ implementation pending'
        }

    def _run_numba_regression(self, X, y, regression_type, function_type):
        """Run regression using Numba implementation."""
        print(f"  Numba implementation not yet available")
        return {
            'status': 'not_implemented',
            'engine': 'numba',
            'message': 'Numba implementation pending'
        }

    def _run_pure_regression(self, X, y, regression_type, function_type):
        """Run regression using Pure Python implementation."""
        print(f"  Pure Python implementation not yet available")
        return {
            'status': 'not_implemented',
            'engine': 'pure',
            'message': 'Pure Python implementation pending'
        }

    def print_results(self):
        """Print summary of all regression results."""
        print(f"\n{S_BOLD}=== Regression Results Summary ==={E_BOLD}")

        successful = 0
        failed = 0
        not_implemented = 0

        for (reg_type, func_type), result in self.results.items():
            reg_name = self.regression_mapping[reg_type]

            if result is None:
                print(f"FAILED: {reg_name} + Function {func_type}")
                failed += 1
            elif result.get('status') == 'not_implemented':
                print(f"NOT IMPLEMENTED: {reg_name} + Function {func_type}")
                not_implemented += 1
            else:
                coeffs = result.get('coefficients', [])
                print(f"SUCCESS: {reg_name} + Function {func_type}: {len(coeffs)} coefficients")
                successful += 1

        print(f"\n{S_BOLD}Summary:{E_BOLD} {successful} successful, {failed} failed, {not_implemented} not implemented")


class VisualizationData:
    """Class for visualizing regression results."""

    def __init__(self, X, y, results):
        self.X = X
        self.y = y
        self.results = results

    def plot_results(self, selected_results=None):
        """Plot regression results."""
        if selected_results is None:
            selected_results = self.results

        # Počet úspěšných výsledků
        successful_results = [(k, v) for k, v in selected_results.items()
                              if v is not None and v.get('status') != 'not_implemented']

        if not successful_results:
            print("No successful results to plot!")
            return

        # Vytvoříme subplot grid
        n_plots = len(successful_results)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Vytvoříme x body pro smooth křivky
        x_smooth = np.linspace(self.X.min(), self.X.max(), 300)

        for idx, ((reg_type, func_type), result) in enumerate(successful_results):
            ax = axes[idx] if n_plots > 1 else axes[0]

            # Plot originální data
            ax.scatter(self.X, self.y, alpha=0.5, label='Data')

            # Získáme predikce
            model = result['model']

            try:
                if func_type >= 8:  # Speciální funkce
                    # Transformujeme x_smooth stejně jako při fitu
                    X_smooth_transformed, _ = self._transform_features_for_prediction(
                        x_smooth.reshape(-1, 1), func_type
                    )
                    y_pred_smooth = model.predict(X_smooth_transformed)

                    # Pro semi-log transformaci musíme výsledek exponenciovat
                    if func_type == 10:
                        y_pred_smooth = np.exp(y_pred_smooth)
                else:
                    # Pro polynomy
                    y_pred_smooth = model.predict(x_smooth)

                ax.plot(x_smooth, y_pred_smooth, 'r-', label='Fit', linewidth=2)

            except Exception as e:
                print(f"Error plotting {reg_type}, {func_type}: {e}")
                continue

            # Nastavení grafu
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

        # Skryjeme prázdné subploty
        for idx in range(len(successful_results), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

    def _transform_features_for_prediction(self, X, function_type):
        """Transformuje features pro predikci podle typu funkce."""
        X = X.flatten()
        min_val = 1e-10

        if function_type == 8:  # Log-Linear
            X_positive = np.where(X > 0, X, min_val)
            X_transformed = np.log(X_positive).reshape(-1, 1)
            return X_transformed, None

        elif function_type == 9:  # Log-Polynomial
            X_positive = np.where(X > 0, X, min_val)
            X_log = np.log(X_positive)
            X_transformed = np.column_stack([X_log, X_log ** 2])
            return X_transformed, None

        elif function_type == 10:  # Semi-Log
            return X.reshape(-1, 1), None

        elif function_type == 11:  # Square Root
            X_positive = np.where(X >= 0, X, 0)
            X_transformed = np.sqrt(X_positive).reshape(-1, 1)
            return X_transformed, None

        elif function_type == 12:  # Inverse
            X_nonzero = np.where(np.abs(X) > min_val, X, min_val * np.sign(X))
            X_nonzero = np.where(X_nonzero == 0, min_val, X_nonzero)
            X_transformed = (1.0 / X_nonzero).reshape(-1, 1)
            return X_transformed, None

        elif function_type == 13:  # Log-Sqrt
            X_positive = np.where(X > 0, X, min_val)
            X_log = np.log(X_positive)
            X_sqrt = np.sqrt(X_positive)
            X_transformed = np.column_stack([X_log, X_sqrt])
            return X_transformed, None

        elif function_type == 14:  # Mixed
            X_positive = np.where(X > 0, X, min_val)
            X_log = np.log(X_positive)
            X_transformed = np.column_stack([X, X_log])
            return X_transformed, None

        elif function_type == 15:  # Poly-Log
            X_positive = np.where(X > 0, X, min_val)
            X_log = np.log(X_positive)
            X_transformed = np.column_stack([X, X ** 2, X_log])
            return X_transformed, None

        elif function_type == 16:  # Volatility Mix
            X_positive = np.where(X > 0, X, min_val)
            X_sqrt = np.sqrt(X_positive)
            X_inv = 1.0 / X_positive
            X_transformed = np.column_stack([X_sqrt, X_inv])
            return X_transformed, None

        else:
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

class TimerRegressionEngines:
    pass


def main():
    # Load or create data.
    data_loader = DataLoader()
    X, y = data_loader.get_data()

    # Print load data format.
    print_data_loaded(X, y)

    # Get user choices.
    input_handler = UserInputHandler()

    engine_choice = input_handler.get_engine_choice()
    regression_types = input_handler.get_regression_types()
    function_types = input_handler.get_function_types()

    print_selected_specifications(engine_choice, regression_types, function_types)

    # Run regressions
    regression_runner = RegressionRun(engine_choice, regression_types, function_types)
    results = regression_runner.run_regressions(X, y)
    regression_runner.print_results()

    # Vizualizace výsledků
    while True:
        user_input = input(
            f"\n{S_BOLD}Would you like to:{E_BOLD}\n1. Visualize results\n2. Print coefficients\n3. Exit\nChoose option (1-3): ")

        if user_input == '1':
            visualizer = VisualizationData(X, y, results)
            visualizer.plot_results()
        elif user_input == '2':
            visualizer = VisualizationData(X, y, results)
            visualizer.print_coefficients()
        elif user_input == '3':
            print("Exiting...")
            break
        else:
            print(f"{S_RED}Invalid input{E_RED}: Please enter 1, 2, or 3")


if __name__ == "__main__":
    main()