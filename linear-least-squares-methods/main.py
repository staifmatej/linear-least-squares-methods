"""Main script for testing linear regression implementations - FIXED VERSION."""


from utils import (
    DataLoader,
    RegressionRun,
    VisualizationData,
    UserInputHandler,
    print_data_loaded,
    print_selected_specifications,
    print_selected_configurations,
    print_condition_numbers,
    print_coefficients
)

# Global constants used for bold text and red warning messages.
S_BOLD = "\033[1m"
E_BOLD = "\033[0m"
S_RED = "\033[91m"
E_RED = "\033[0m"


def show_results_menu(X, y, results, engine_choice, regression_types, function_types):
    """Show results menu and handle user choices."""
    while 1:
        print("\n══════ Would you like to ══════")
        print("1. Visualize results on one image")
        print("2. Visualize results per image")
        print("3. Print coefficients")
        print("4. Print condition numbers of methods")
        print("5. Print selected configurations")
        print("6. Exit")
        print("═══════════════════════════════\n")

        user_input = input("Choose option (1-6): ")

        if user_input == '1':
            visualizer = VisualizationData(X, y, results)
            visualizer.plot_results()
        elif user_input == '2':
            # Visualize results per image (individual plots)
            visualizer = VisualizationData(X, y, results)
            visualizer.plot_results_individually()
        elif user_input == '3':
            print_coefficients(results, regression_types, function_types)
        elif user_input == '4':
            print_condition_numbers(results, regression_types, function_types)
        elif user_input == '5':
            print_selected_configurations(engine_choice, regression_types, function_types)
        elif user_input == '6':
            break
        else:
            print(f"{S_RED}Invalid input{E_RED}: Please enter 1, 2, 3, 4, 5, or 6")


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

    # Print selected specifications
    print_selected_specifications(engine_choice, regression_types, function_types)

    # Run regressions
    regression_runner = RegressionRun(engine_choice, regression_types, function_types)
    results = regression_runner.run_regressions(X, y)
    regression_runner.print_results()

    # Show results menu
    show_results_menu(X, y, results, engine_choice, regression_types, function_types)


if __name__ == "__main__":
    main()
