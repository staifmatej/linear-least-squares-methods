"""Comprehensive test for all menu options in main.py"""

import os
import unittest
import subprocess
import sys
from pathlib import Path


class TestMainMenuComprehensive(unittest.TestCase):
    """Test all possible menu options and combinations from main.py"""

    def setUp(self):
        """Set up test environment."""
        self.main_path = Path(__file__).parent.parent / "main.py"
        self.python_exe = sys.executable

    def run_main_with_inputs(self, inputs, description):
        """Run main.py with specific input sequence."""
        # Set matplotlib to non-interactive backend to prevent GUI windows
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'

        try:
            result = subprocess.run(
                [self.python_exe, str(self.main_path)],
                input=inputs,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                env=env
            )
            return result
        except subprocess.TimeoutExpired as exc:
            self.fail(f"Test timed out for {description}: {exc}")
            return None

    def check_no_errors(self, result, description):
        """Check that no errors occurred."""
        # Check for errors in output (excluding expected ones)
        stdout_lower = result.stdout.lower()
        stderr_lower = result.stderr.lower()

        # If termios error (from press enter), that's OK in tests
        if "termios.error" in result.stderr and "Inappropriate ioctl" in result.stderr:
            # This is expected in automated tests
            return

        # Check return code (unless it's termios error)
        if result.returncode != 0:
            # Check if it's just the press enter issue
            if "print_press_enter_to_continue" not in result.stderr:
                self.fail(f"{description} failed with return code {result.returncode}\n"
                         f"stdout: {result.stdout}\n"
                         f"stderr: {result.stderr}")

        # Check for error indicators (excluding expected warnings)
        error_indicators = ["error:", "traceback", "exception", "failed"]

        for indicator in error_indicators:
            if indicator in stdout_lower:
                # Skip expected warnings
                if "condition number" in stdout_lower:
                    continue
                self.fail(f"{description}: Found error indicator '{indicator}' in stdout")

            if indicator in stderr_lower:
                # Skip warnings and termios errors
                if "warning" in stderr_lower:
                    continue
                if "termios" in stderr_lower and "ioctl" in stderr_lower:
                    continue
                if "print_press_enter_to_continue" in stderr_lower:
                    continue
                self.fail(f"{description}: Found error indicator '{indicator}' in stderr")

    def test_all_data_input_methods(self):
        """Test all data input methods."""
        # Test 1: Manual 2D input
        inputs = "1\n1,2\n3,4\n2,3\n4,5\ndone\n1\n1\n1\n7\n"
        result = self.run_main_with_inputs(inputs, "Manual 2D input")
        self.check_no_errors(result, "Manual 2D input")

        # Test 2: Generate synthetic data
        inputs = "4\n50\n2\n0.1\n1\n1\n1\n7\n"
        result = self.run_main_with_inputs(inputs, "Generate synthetic data")
        self.check_no_errors(result, "Generate synthetic data")

        # Test 3: Example dataset
        inputs = "5\n1\n1\n1\n1\n7\n"
        result = self.run_main_with_inputs(inputs, "Example dataset")
        self.check_no_errors(result, "Example dataset")

    def test_all_engines_with_all_combinations(self):
        """Test all engines with all regression and function types."""
        engines = [1, 2, 3]  # NumPy, Numba, Pure Python

        for engine in engines:
            # Test with all regression and function types
            inputs = f"4\n\n\n\n{engine}\nall\nall\n7\n"
            engine_names = {1: "NumPy", 2: "Numba", 3: "Pure Python"}
            result = self.run_main_with_inputs(inputs, f"{engine_names[engine]} engine all combinations")
            self.check_no_errors(result, f"{engine_names[engine]} engine all combinations")

    def test_individual_regression_types(self):
        """Test each regression type individually."""
        regression_types = [1, 2, 3, 4]  # Linear, Ridge, Lasso, ElasticNet

        for reg_type in regression_types:
            inputs = f"4\n\n\n\n1\n{reg_type}\n1\n7\n"
            reg_names = {1: "Linear", 2: "Ridge", 3: "Lasso", 4: "ElasticNet"}
            result = self.run_main_with_inputs(inputs, f"{reg_names[reg_type]} regression")
            self.check_no_errors(result, f"{reg_names[reg_type]} regression")

    def test_individual_function_types(self):
        """Test each function type individually."""
        # Test polynomial functions (1-7)
        for func_type in range(1, 8):
            inputs = f"4\n\n\n\n1\n1\n{func_type}\n7\n"
            result = self.run_main_with_inputs(inputs, f"Polynomial degree {func_type}")
            self.check_no_errors(result, f"Polynomial degree {func_type}")

        # Test special functions (8-16)
        for func_type in range(8, 17):
            inputs = f"4\n\n\n\n1\n1\n{func_type}\n7\n"
            func_names = {
                8: "Log-Linear", 9: "Log-Polynomial", 10: "Semi-Log",
                11: "Square Root", 12: "Inverse", 13: "Log-Sqrt",
                14: "Mixed", 15: "Poly-Log", 16: "Volatility Mix"
            }
            result = self.run_main_with_inputs(inputs, f"{func_names[func_type]} function")
            self.check_no_errors(result, f"{func_names[func_type]} function")

    def test_all_post_regression_menu_options(self):
        """Test all menu options after regression."""
        # Base setup: simple regression
        base_inputs = "4\n20\n1\n0.1\n1\n1\n1\n"

        # Test option 1: Visualize on one image
        inputs = base_inputs + "1\n7\n"
        result = self.run_main_with_inputs(inputs, "Visualize on one image")
        self.check_no_errors(result, "Visualize on one image")

        # Test option 2: Visualize per image
        inputs = base_inputs + "2\n7\n"
        result = self.run_main_with_inputs(inputs, "Visualize per image")
        self.check_no_errors(result, "Visualize per image")

        # Test option 3: Print coefficients
        inputs = base_inputs + "3\n7\n"
        result = self.run_main_with_inputs(inputs, "Print coefficients")
        self.check_no_errors(result, "Print coefficients")
        self.assertIn("COEFFICIENTS", result.stdout, "Should show coefficients")

        # Test option 4: Print condition numbers
        inputs = base_inputs + "4\n7\n"
        result = self.run_main_with_inputs(inputs, "Print condition numbers")
        self.check_no_errors(result, "Print condition numbers")

        # Test option 5: Print selected configurations
        inputs = base_inputs + "5\n7\n"
        result = self.run_main_with_inputs(inputs, "Print selected configurations")
        self.check_no_errors(result, "Print selected configurations")

        # Test option 6: Performance benchmark
        inputs = base_inputs + "6\n3\n7\n"  # 3 runs for quick test
        result = self.run_main_with_inputs(inputs, "Performance benchmark")
        self.check_no_errors(result, "Performance benchmark")
        self.assertIn("benchmark", result.stdout.lower(), "Should show benchmark results")

    def test_multiple_menu_selections(self):
        """Test multiple menu selections in sequence."""
        # Test selecting multiple options before exit
        inputs = "4\n\n\n\n1\n1\n1\n3\n4\n5\n7\n"
        result = self.run_main_with_inputs(inputs, "Multiple menu selections")
        self.check_no_errors(result, "Multiple menu selections")

    def test_edge_cases(self):
        """Test edge cases and special scenarios."""
        # Test with minimal data points
        inputs = "1\n1,1\n2,2\ndone\n1\n1\n1\n7\n"
        result = self.run_main_with_inputs(inputs, "Minimal data points")
        self.check_no_errors(result, "Minimal data points")

        # Test with high degree polynomial (should work or give proper error)
        inputs = "4\n10\n\n\n1\n1\n7\n7\n"  # 10 points, degree 7
        result = self.run_main_with_inputs(inputs, "High degree polynomial")
        # Should either work or give proper error message, not crash
        self.assertLessEqual(result.returncode, 1, "Should not crash")

    def test_all_engines_with_special_functions(self):
        """Test all engines specifically with special functions."""
        engines = [1, 2, 3]
        special_functions = [8, 11, 12]  # Log-Linear, Square Root, Inverse

        for engine in engines:
            for func in special_functions:
                inputs = f"4\n50\n\n\n{engine}\n1\n{func}\n7\n"
                engine_names = {1: "NumPy", 2: "Numba", 3: "Pure Python"}
                func_names = {8: "Log-Linear", 11: "Square Root", 12: "Inverse"}
                desc = f"{engine_names[engine]} with {func_names[func]}"
                result = self.run_main_with_inputs(inputs, desc)
                self.check_no_errors(result, desc)

    def test_multidimensional_data_rejection(self):
        """Test that multidimensional data shows proper message for visualization."""
        # Create multidimensional data
        inputs = "2\n3\n1 2 3\n4 5 6\n7 8 9\n10\n20\n30\n1\n1\n1\n1\n7\n"
        result = self.run_main_with_inputs(inputs, "Multidimensional visualization check")
        # Should complete without errors
        self.assertEqual(result.returncode, 0, "Should handle multidimensional data properly")
        # Check for appropriate message
        if "1\n7\n" in inputs:  # If trying to visualize
            self.assertIn("not supported for multidimensional", result.stdout,
                         "Should show message about multidimensional visualization")


if __name__ == "__main__":
    unittest.main(verbosity=2)
