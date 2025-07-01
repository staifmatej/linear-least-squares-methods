"""Post-regression results handler with interactive UI components, result formatting, and condition number analysis."""

# for function print_press_enter_to_continue()
import sys
import select
import tty
import termios

from constants import S_BOLD, S_RED, E_BOLD, E_RED


def print_press_enter_to_continue():
    """Function print message to console with effects of loading dots."""
    # Check if running in Jupyter/IPython
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook
            input("\nPress Enter to continue...")
            return
    except NameError:
        pass  # Not in IPython/Jupyter
    try:
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())

        dots = ["   ", ".  ", ".. ", "..."]
        dot_index = 0

        print("\n")
        while 1:
            # Check if Enter was pressed
            if select.select([sys.stdin], [], [], 0.5)[0]:
                char = sys.stdin.read(1)
                if ord(char) == 13:  # Enter key
                    break

            # Update dots animation
            sys.stdout.write(f"\rPress Enter to continue{dots[dot_index]}")
            sys.stdout.flush()
            dot_index = (dot_index + 1) % len(dots)

        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\rPress Enter to continue...    ")

    except (ImportError, OSError, RuntimeError):
        # Fallback for systems without select/termios
        input("\nPress Enter to continue...")


def print_data_loaded(X, y):
    """Print loaded data dimensions."""
    print("\n═════════ DATA LOADED ═════════")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("═══════════════════════════════\n")


def print_selected_specifications(engine_choice, regression_types, function_types):
    """Print selected configuration summary."""
    engine_mapping = {
        1: "approaches/least_squares_numpy.py",
        2: "approaches/least_squares_numba.py",
        3: "approaches/least_squares_pure.py"
    }

    print("\n═══ SELECTED CONFIGURATION ════")
    print(f"Engine: {engine_mapping.get(engine_choice)}")
    print(f"Regression types: {regression_types}")
    print(f"Function types: {function_types}")
    print("═══════════════════════════════\n")


def print_selected_configurations(engine_choice, regression_types, function_types):
    """Print detailed selected configuration for menu option 5."""
    engine_mapping = {
        1: "Python with NumPy",
        2: "Python with Numba",
        3: "Pure Python with for-loops"
    }

    regression_mapping = {
        1: "Linear regression (Least Squares)",
        2: "Ridge regression (Least Squares)",
        3: "Lasso Regression (Coordinate Descent)",
        4: "Elastic Net Regression (Coordinate Descent)"
    }

    function_mapping = {
        1: "Linear (y = a + b*x)",
        2: "Quadratic (y = a + b*x + c*x^2)",
        3: "Cubic (y = a + b*x + c*x^2 + d*x^3)",
        4: "Quartic (y = a + b*x + c*x^2 + d*x^3 + e*x^4)",
        5: "Quintic (y = a + b*x + c*x^2 + ... + e*x^4 + f*x^5)",
        6: "Sextic (y = a + b*x + c*x^2 + ... + f*x^5 + g*x^6)",
        7: "Septic (y = a + b*x + ... + h*x^7)",
        8: "Log-Linear (y = a + b*log(x))",
        9: "Log-Polynomial (y = a + b*log(x) + c*log(x)^2)",
        10: "Semi-Log (log(y) = a + b*x)",
        11: "Square Root (y = a + b*sqrt(x))",
        12: "Inverse (y = a + b/x)",
        13: "Log-Sqrt (y = a + b*log(x) + c*sqrt(x))",
        14: "Mixed (y = a + b*x + c*log(x))",
        15: "Poly-Log (y = a + b*x + c*x^2 + d*log(x))",
        16: "Volatility Mix (y = a + b*sqrt(x) + c/x)"
    }

    print("\n════ CONFIGURATION DETAILS ════")

    # Engine information
    print(f"{S_BOLD}Regression Engine:{E_BOLD}")
    print(f"   {engine_mapping[engine_choice]}")

    # Regression types
    print(f"\n{S_BOLD}Regression Types:{E_BOLD}")
    for reg_type in regression_types:
        print(f"   • {regression_mapping[reg_type]}")

    # Function types
    print(f"\n{S_BOLD}Function Types:{E_BOLD}")
    for func_type in function_types:
        print(f"   • {function_mapping[func_type]}")

    # Summary
    total_combinations = len(regression_types) * len(function_types)
    print(f"\n{S_BOLD}Summary:{E_BOLD}")
    print(f"   • Total combinations: {total_combinations}")
    print(f"   • Regression methods: {len(regression_types)}")
    print(f"   • Function types: {len(function_types)}")

    print("═══════════════════════════════")

    print_press_enter_to_continue()


def print_condition_numbers(results, regression_types, function_types):  # pylint: disable=too-many-locals
    """Print condition numbers for all fitted models."""
    regression_mapping = {
        1: "Linear regression",
        2: "Ridge regression",
        3: "Lasso regression",
        4: "ElasticNet regression"
    }

    function_mapping = {
        1: "Linear", 2: "Quadratic", 3: "Cubic", 4: "Quartic", 5: "Quintic", 6: "Sextic", 7: "Septic",
        8: "Log-Linear", 9: "Log-Polynomial", 10: "Semi-Log", 11: "Square Root", 12: "Inverse",
        13: "Log-Sqrt", 14: "Mixed", 15: "Poly-Log", 16: "Volatility Mix"
    }

    def get_condition_status(cond_num):
        """Inner function to get color and status for condition number."""
        if cond_num >= 1e15:
            return S_RED, "SINGULAR/EXTREMELY ILL-CONDITIONED"
        if cond_num >= 1e13:
            return S_RED, "POORLY CONDITIONED"
        if cond_num >= 1e10:
            return "\033[93m", "MODERATELY CONDITIONED"  # Yellow
        return "\033[92m", "WELL CONDITIONED"  # Green

    def extract_condition_number(result):
        """Inner function to extract condition number from result."""
        condition_num = None
        model = result.get('model')
        if model and hasattr(model, 'condition_number') and model.condition_number is not None:
            condition_num = model.condition_number
        elif result.get('condition_number') is not None:
            condition_num = result.get('condition_number')
        return condition_num

    def print_condition_result(reg_type, func_type, condition_num):
        """Inner function to print condition number results."""
        reg_name = regression_mapping[reg_type]
        func_name = function_mapping[func_type]
        color, status = get_condition_status(condition_num)

        print(f"\n{S_BOLD}{reg_name} - {func_name}:{E_BOLD}")
        print(f"   Condition number: {color}{condition_num:.2e}{E_RED}")
        print(f"   Status: {color}{status}{E_RED}")

    def print_na_result(reg_type, func_type):
        """Inner function to print N/A results."""
        reg_name = regression_mapping[reg_type]
        func_name = function_mapping[func_type]
        print(f"\n{S_BOLD}{reg_name} - {func_name}:{E_BOLD}")
        print(f"   Condition number: {S_RED}N/A (not computed for this method){E_RED}")

    print("\n═════ CONDITION NUMBERS ══════")

    found_any = False
    for reg_type in regression_types:
        for func_type in function_types:
            result = results.get((reg_type, func_type))
            if result is not None and result.get('status') not in ['not_implemented', 'failed', 'not_available']:
                condition_num = extract_condition_number(result)

                if condition_num is not None:
                    print_condition_result(reg_type, func_type, condition_num)
                    found_any = True
                elif result:
                    print_na_result(reg_type, func_type)
                    found_any = True

    if not found_any:
        print(f"\n{S_RED}No condition numbers available.{E_RED}")
        print("This can happen if:")
        print("  • No models were successfully fitted")
        print("  • All selected methods don't compute condition numbers")

    print("═══════════════════════════════")

    print("\nNote: Condition numbers are computed for Polynomial and Ridge regression only.")
    print("Lasso and ElasticNet use coordinate descent and don't require condition number analysis.")

    print_press_enter_to_continue()

def print_coefficients(results, regression_types, function_types):
    """Print coefficients for all successful results with enhanced formatting."""

    print("\n═══ REGRESSION COEFFICIENTS ═══")

    regression_names = {1: "Linear", 2: "Ridge", 3: "Lasso", 4: "ElasticNet"}
    function_names = {
        1: "Linear", 2: "Quadratic", 3: "Cubic", 4: "Quartic",
        5: "Quintic", 6: "Sextic", 7: "Septic",
        8: "Log-Linear", 9: "Log-Polynomial", 10: "Semi-Log",
        11: "Square Root", 12: "Inverse", 13: "Log-Sqrt",
        14: "Mixed", 15: "Poly-Log", 16: "Volatility Mix"
    }

    found_any = False
    for reg_type in regression_types:
        for func_type in function_types:
            result = results.get((reg_type, func_type))
            if result is None:
                continue

            # Handle different result formats
            if result.get('status') in ['not_implemented', 'failed', 'not_available']:
                continue

            coeffs = result.get('coefficients', [])
            if coeffs is None or (hasattr(coeffs, '__len__') and len(coeffs) == 0):
                continue

            found_any = True

            # Model name header
            model_name = f"{regression_names[reg_type]} - {function_names.get(func_type, f'Function {func_type}')}"
            print(f"{S_BOLD}{model_name}{E_BOLD}")

            # Intercept
            print(f"   Intercept:      {coeffs[0]:.6f}")

            # Coefficients
            for i, coeff in enumerate(coeffs[1:], 1):
                if i <= 3:
                    coeff_name = f"β_{i}"
                    print(f"   {coeff_name}:            {coeff:.6f}")
                else:
                    coeff_name = f"Coefficient {i}"
                    print(f"   {coeff_name}:  {coeff:.6f}")

    if not found_any:
        print("\nNo coefficients available to display.")

    print("═══════════════════════════════")
    print_press_enter_to_continue()
