"""Main script for testing polynomial regression implementations - FIXED VERSION."""

from utils import DataLoader, RegressionRun, VisualizationData, UserInputHandler

# Global constants used for bold text and red warning messages.
S_BOLD = "\033[1m"
E_BOLD = "\033[0m"
S_RED = "\033[91m"
E_RED = "\033[0m"


def print_data_loaded(X, y):
    """Print loaded data dimensions."""
    print("\n========= Data loaded =========")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("===============================\n")


def print_selected_specifications(engine_choice, regression_types, function_types):
    """Print selected configuration summary."""
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


def main():
    """Main entry point for the interactive regression testing environment."""
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

    # Visualization of results
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
