"""Data loading utilities for various input sources."""

import os
import numpy as np
import pandas as pd

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
        print("\n═════ Data Input Options ═════")
        print("1. Manual input (2D points)")
        print("2. Manual input (multidimensional)")
        print("3. Load from CSV file")
        print("4. Generate synthetic data")
        print("5. Use example dataset")
        print("════════════════════════════════")

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


    # pylint: disable=too-many-branches
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
        y_col = input("Enter target column name (y-axis): ").strip()
        data.columns = data.columns.str.strip()
        if y_col not in data.columns:
            print(f"{S_RED}Warning{E_RED}: Column not found!")
        # Select feature columns - limit to single column for 2D visualization
        print("Enter ONE feature column name for 2D visualization:")
        print("Available numeric columns:", [col for col in data.columns if col != y_col and data[col].dtype != 'object'])
        x_cols_input = input("Feature column (x-axis): ").strip()

        if x_cols_input:
            x_cols = [x_cols_input.strip()]
        else:
            # Default to first numeric column
            numeric_cols = [col for col in data.columns if col != y_col and data[col].dtype != 'object']
            if numeric_cols:
                x_cols = [numeric_cols[0]]
            else:
                raise ValueError("No numeric columns available for features")

        # Convert to numeric, handling non-numeric columns
        X_numeric = []
        for col in x_cols:
            if data[col].dtype == 'object':
                # Try to convert to numeric, skip if not possible
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    if data[col].isna().all():
                        print(f"{S_RED}Warning{E_RED}: Column '{col}' contains no numeric data, skipping...")
                        continue
                except (ValueError, TypeError, pd.errors.ParserError):
                    print(f"{S_RED}Warning{E_RED}: Column '{col}' cannot be converted to numeric, skipping...")
                    continue
            X_numeric.append(col)

        if not X_numeric:
            raise ValueError("No numeric columns found for features")

        X_csv = data[X_numeric].values
        y_csv = data[y_col].values

        print(f"\nSelected {len(X_numeric)} features: {X_numeric}")
        print(f"Target: {y_col}")

        return X_csv, y_csv

    def _csv_files_not_found(self, filename, csv_files):
        """Display CSV file selection menu and get user choice."""
        if len(csv_files) == 1:
            print("════ Found CSV file option ═════")
        else:
            print("═══ Found CSV files options ════")
        print(f"Found {len(csv_files)} CSV files:")
        for i, file in enumerate(csv_files, 1):
            print(f"{i}. {file}")
        print("════════════════════════════════")

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
                if n_samples > 1e8:
                    print(f"{S_RED}Invalid input{E_RED}: Number of samples too large (max. 1e8)")
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
                    print(f"{S_RED}Invalid input{E_RED}: Polynomial degree too extremely high (max. 100)")
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

        print(f"\nGenerated {n_samples} points with degree {poly_degree_synthetic} polynomial")
        print(f"True coefficients: {true_coeffs}")

        return x_synthetic.reshape(-1, 1), y_synthetic_noisy

    def use_example_dataset(self):
        """Use predefined example dataset."""
        print("\n═══════ Example datasets ═══════")
        print("1. House prices (size vs price)")
        print("2. Temperature data (time vs temp)")
        print("3. Quadratic function")
        print("4. Sinusoidal function")
        print("════════════════════════════════\n")
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

        if choice == 2:
            # Temperature data
            x_time = np.linspace(0, 24, 25)  # 24 hours
            y_temperature = 15 + 10 * np.sin(2 * np.pi * x_time / 24) + np.random.normal(0, 1, 25)
            return x_time.reshape(-1, 1), y_temperature

        if choice == 3:
            # Quadratic function
            x_quadratic = np.linspace(-3, 3, 20)
            y_quadratic = 2 * x_quadratic ** 2 + 3 * x_quadratic + 1 + np.random.normal(0, 2, 20)
            return x_quadratic.reshape(-1, 1), y_quadratic

        # choice == 4; Sinusoidal function
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
