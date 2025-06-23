"""User input handling utilities for regression and function type selection."""

# Global constants used for bold text and red warning messages.
S_BOLD = "\033[1m"
E_BOLD = "\033[0m"
S_RED = "\033[91m"
E_RED = "\033[0m"

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
            user_input = input(
                f"Enter types of regression you would you like to try: (for example: {S_BOLD}all{E_BOLD} or {S_BOLD}1,2,3{E_BOLD} or {S_BOLD}4{E_BOLD}): ").strip()

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

                raise ValueError(f"{S_RED}Invalid input{E_RED}: No valid choices!")

            except ValueError:
                print(
                    f"{S_RED}Invalid input{E_RED}: Please enter numbers 1-4 separated by commas or '{S_BOLD}all{E_BOLD}'")
                print(f"Example: {S_BOLD}1,2,3{E_BOLD} or {S_BOLD}all{E_BOLD}")

    def get_function_types(self):
        """Get function types selection from user."""

        print("\n==== Types of function fit ====")
        print("1. Linear (y = a + b*x)")  # Polynomial of degree 1.
        print("2. Quadratic (y = a + b*x + c*x^2)")  # Polynomial of degree 2.
        print("3. Cubic (y = a + b*x + c*x^2 + d*x^3)")  # Polynomial of degree 3.
        print("4. Quartic (y = a + b*x + c*x^2 + d*x^3 + e*x^4)")  # Polynomial of degree 4.
        print("5. Quintic (y = a + b*x + c*x^2 + ... + e*x^4 + f*x^5)")  # Polynomial of degree 5.
        print("6. Sextic (y = a + b*x + c*x^2 + ... + f*x^5 + g*x^6)")  # Polynomial of degree 6.
        print("7. Septic (y = a + b*x + ... + h*x^7)")  # Polynomial of degree 7.
        print("8. Log-Linear (y = a + b*log(x))")  # Interest rate curves.
        print("9. Log-Polynomial (y = a + b*log(x) + c*log(x)^2)")  # Volatility smile.
        print("10. Semi-Log (log(y) = a + b*x)")  # Exponential growth.
        print("11. Square Root (y = a + b*sqrt(x))")  # VIX, volatility √time.
        print("12. Inverse (y = a + b/x)")  # Mean reversion speed.
        print("13. Log-Sqrt (y = a + b*log(x) + c*sqrt(x))")  # Complex volatility.
        print("14. Mixed (y = a + b*x + c*log(x))")  # Yield curves.
        print("15. Poly-Log (y = a + b*x + c*x^2 + d*log(x))")  # Nelson-Siegel like.
        print("16. Volatility Mix (y = a + b*sqrt(x) + c/x)")  # GARCH approximation.
        print("===============================\n")

        while 1:
            user_input = input(
                f"Enter types of functions you would you like to fit: (for example: {S_BOLD}all{E_BOLD} or {S_BOLD}1,2,3{E_BOLD} or {S_BOLD}9{E_BOLD}): ").strip()

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

                raise ValueError(f"{S_RED}Invalid input{E_RED}: No valid choices")

            except ValueError:
                print(
                    f"{S_RED}Invalid input{E_RED}: Please enter numbers 1-16 separated by commas or '{S_BOLD}all{E_BOLD}'")
                print(f"Example: {S_BOLD}1,2,3{E_BOLD} or {S_BOLD}all{E_BOLD}")
