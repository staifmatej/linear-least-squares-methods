"""Main script for testing polynomial regression implementations."""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from approaches.least_squares_numpy import LeastSquares, PolynomialRegression

# Global constants used for bold text and red warning messages.
S_BOLD = "\033[1m"
E_BOLD = "\033[0m"
S_RED = "\033[91m"
E_RED = "\033[0m"

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
                print("Please enter number 1-5")
            except ValueError:
                print("Please enter a valid number")

    def input_2d_points(self):
        """Input 2D points manually."""

        print("\nEnter 2D points [x,y]. Type 'done' when finished.")
        print("Example: 1,2.5")

        x_data, y_data = [], []

        point_count = 0
        while 1:
            user_input = input("x,y: ").strip()
            if user_input.lower() == 'done':
                if point_count <= 2:
                    print("\nNeed at least 3 points!")
                    continue
                break

            try:
                x_2d_point, y_2d_point = map(float, user_input.split(','))
                x_data.append(x_2d_point)
                y_data.append(y_2d_point)
                print(f"Added point: [{x_2d_point}, {y_2d_point}]")
                point_count += 1
            except ValueError:
                print("Invalid format! Use: x,y (example: 1.5,2.3)")

        return np.array(x_data).reshape(-1, 1), np.array(y_data)

    def input_multidimensional(self):
        """Input multidimensional data manually."""
        print("\nMultidimensional data input")

        while 1:
            try:
                n_features = int(input("How many features (X dimensions): "))
                if n_features >= 1:
                    break
                print("Must be at least 1 feature")
            except ValueError:
                print("Please enter a valid number")

        print(f"\nEnter data points with {n_features} features + 1 target value")
        print(f"Format: x₁,x₂,...,x{n_features},y")
        print("Type 'done' when finished")

        data_points = []

        while True:
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
                print("Invalid format! Use numbers separated by commas")

        if len(data_points) < 2:
            print("Need at least 2 points!")
            return self.input_multidimensional()

        data_array = np.array(data_points)
        X_multidim = data_array[:, :-1]  # All columns except last
        y_multidim = data_array[:, -1]  # Last column

        return X_multidim, y_multidim

    def _find_csv_files(self):
        """Find CSV files recursively in all subdirectories using os.walk()."""
        csv_files = []

        print(f"\n{S_BOLD}Searching for CSV files recursively...{E_BOLD}\n")
        for root, files in os.walk('.'):
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
            print(f"\nError: {filename} is empty!")
            return self.load_from_csv()

        # Check minimum requirements
        if data.shape[0] < 2:
            print(f"\nError: {filename} has only {data.shape[0]} rows. Need at least 2 rows!")
            return self.load_from_csv()

        if data.shape[1] < 2:
            print(f"\nError: {filename} has only {data.shape[1]} columns. Need at least 2 columns!")
            return self.load_from_csv()

        print(f"\nLoaded {filename} with shape: {data.shape}")
        print("First few rows:")
        print(data.head())

        print("\nColumn selection:")
        print("Columns:", list(data.columns))

        # Select target column
        y_col = input("Enter target column name (y): ").strip()
        if y_col not in data.columns:
            print("Column not found!")
            return self.load_from_csv()

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
        csv_files = self._find_csv_files()

        if not csv_files:
            print("\nNo CSV files found in current directory or subdirectories!")
            filename = input("Enter full path to CSV file: ")
        else:
            filename = None
            filename, csv_files = self._csv_files_not_found(filename, csv_files)

        try:
            X_csv, y_csv = self._csv_files_read_data(filename)
            return X_csv, y_csv
        except FileNotFoundError:
            print(f"File {filename} not found!")
            return self.load_from_csv()
        except (OSError, ValueError, KeyError) as e:
            print(f"Error loading file: {e}")
            return self.load_from_csv()

    def generate_synthetic_data(self):
        """Generate synthetic polynomial data."""
        print("\nSynthetic data generation")

        try:
            n_samples = int(input("Number of samples (default 50): ") or "50")
            poly_degree_synthetic = int(input("True polynomial degree (default 3): ") or "3")
            noise_level = float(input("Noise level (default 0.1): ") or "0.1")
        except ValueError:
            print("Using default values")
            n_samples, degree, noise_level = 50, 3, 0.1

        np.random.seed(42)
        x_synthetic = np.linspace(-2, 2, n_samples)

        # Generate random coefficients
        true_coeffs = np.random.randn(degree + 1)

        # Generate polynomial
        y_synthetic_true = sum(coeff * x_synthetic ** i for i, coeff in enumerate(true_coeffs))
        y_synthetic_noisy = y_synthetic_true + noise_level * np.random.randn(n_samples)

        print(f"Generated {n_samples} points with degree {degree} polynomial")
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

        choice = input("Choose example (1-4): ").strip()

        if choice == "1":
            # House prices example
            x_house_size = np.array([50, 80, 100, 120, 150, 180, 200, 250, 300])
            y_house_price = np.array([150, 200, 240, 280, 320, 350, 380, 450, 500])
            return x_house_size.reshape(-1, 1), y_house_price

        if choice == "2":
            # Temperature data
            x_time = np.linspace(0, 24, 25)  # 24 hours
            y_temperature = 15 + 10 * np.sin(2 * np.pi * x_time / 24) + np.random.normal(0, 1, 25)
            return x_time.reshape(-1, 1), y_temperature

        if choice == "3":
            # Quadratic function
            x_quadratic = np.linspace(-3, 3, 20)
            y_quadratic = 2 * x_quadratic ** 2 + 3 * x_quadratic + 1 + np.random.normal(0, 2, 20)
            return x_quadratic.reshape(-1, 1), y_quadratic

        # Default sinusoidal (choice == "4")
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


if __name__ == "__main__":
    print(f"{S_BOLD}=== Linear Least Squares Methods Demo ==={E_BOLD}")

    # Load or create data.
    data_loader = DataLoader()
    X, y = data_loader.get_data()

    print("===== Regression Engines =====")
    print("1. C++ with ML Pack")
    print("2. Python with Numpy")
    print("3. Python with Numba")
    print("4. Pure Python with for-loops")
    print(f"{S_BOLD}* Lasso/ElasticNet use sklearn coordinate descent{E_BOLD}")
    print("===============================")

    engine_choice = int(input("\nChoose option (1-4): "))
    # + print napovedu pri spatnem pokusu ze vytiskni opet example co se ma zadat

    print("\n===== Types of regression =====\n")
    print("1. Polynomial regression (use Least Squares)")
    print("2. Ridge regression (use Least Squares)")
    print("3. Lasso Regression (use CoordinateDescent)")
    print("4. Elastic Net Regression (use CoordinateDescent)")
    print("===============================\n")

    input(f"Enter types of regression you would you like to try: (for example: {S_BOLD}all{E_BOLD} or {S_BOLD}1,2,3{E_BOLD} or {S_BOLD}4{E_BOLD}): ")
    # + print napovedu pri spatnem pokusu ze vytiskni opet example co se ma zadat

    print("==== Types of function fit ====")
    print("1. Linear (y = a + b*x)")
    print("2. Quadratic (y = a + b*x + c*x^2)")
    print("3. Cubic (y = a + b*x + c*x^2 + d*x^3)")
    print("4. Quartic (y = a + b*x + c*x^2 + d*x^3 + e*x^4)")
    print("5. Exponential (y = a * e^(b*x))")
    print("6. Logarithmic (y = a + b*log(x))")
    print("7. Power Law (y = a * x^b)")
    print("8. Sigmoid/Logistic (y = L/(1 + e^(-k*(x-x0))))")
    print("9. Volatility (y = a + b*sqrt(x))")
    print("10. Mean Reversion (y = a + b*e^(-c*x) + d)")
    print("11. GARCH-like (y = a + b*x + c*x^2*e^(-d*x))")
    print("12. Interest Rate (y = a*(1-e^(-b*x))/x)")
    print("===============================\n")

    input(f"Enter types of functions you would you like to fit: (for example: {S_BOLD}all{E_BOLD} or {S_BOLD}1,2,3{E_BOLD} or {S_BOLD}9{E_BOLD}): ")

    print(f"\n{S_BOLD}Data loaded:{E_BOLD}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")