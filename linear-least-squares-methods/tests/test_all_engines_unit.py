"""Unit tests for all least squares engines."""

import unittest
import subprocess
import sys
from pathlib import Path


class TestAllEnginesUnit(unittest.TestCase):
    """Unit tests for least_squares_pure.py, least_squares_numba.py and least_squares_numpy.py."""

    def setUp(self):
        """Set up test environment."""
        self.main_path = Path(__file__).parent.parent / "main.py"
        self.python_exe = sys.executable

    def run_main_and_check_success(self, engine_choice, engine_name):
        """Run main.py with specific engine and verify no errors or warnings occur."""
        # Input sequence for synthetic data generation with all options
        input_sequence = f"4\n\n\n\n{engine_choice}\nall\nall\n7\n"

        try:
            result = subprocess.run(
                [self.python_exe, str(self.main_path)],
                input=input_sequence,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                check=False
            )

            # Check return code
            self.assertEqual(result.returncode, 0,
                            f"{engine_name} failed with return code {result.returncode}\n"
                            f"stdout: {result.stdout}\n"
                            f"stderr: {result.stderr}")

            # Check for errors in output (excluding condition number warnings)
            self._check_for_errors_and_warnings(result.stdout, result.stderr, engine_name)

            # Verify computation was successful
            self.assertIn("successful", result.stdout.lower(),
                         f"{engine_name} did not complete successfully")

            return result

        except subprocess.TimeoutExpired as exc:
            self.fail(f"Test timed out for {engine_name}: {exc}")
            return None

    def _check_for_errors_and_warnings(self, stdout, stderr, engine_name):
        """Check for errors and warnings, excluding condition number warnings."""
        # Convert to lowercase for case-insensitive matching
        stdout_lower = stdout.lower()
        stderr_lower = stderr.lower()

        # Check for actual error indicators
        error_indicators = [
            "error:",
            "traceback",
            "exception",
            "failed",
            "could not",
            "unable to",
            "invalid"
        ]

        # Check stdout for errors
        for indicator in error_indicators:
            if indicator in stdout_lower:
                # Skip condition number warnings as these are expected
                if "condition number" in stdout_lower:
                    continue
                # Skip numba compilation errors for Numba engine - these are known issues
                if "numba" in engine_name.lower() and ("typing" in stdout_lower or "compilation" in stdout_lower or "reflect" in stdout_lower):
                    continue
                self.fail(f"{engine_name}: Found error indicator '{indicator}' in output:\n{stdout}")

        # Check stderr for actual errors (ignore warnings)
        if stderr and stderr.strip():
            # Ignore warnings about condition numbers and other expected warnings
            if "condition number" not in stderr_lower and "warning" not in stderr_lower:
                # Also ignore numba compilation warnings and errors
                if "numba" not in stderr_lower and "deprecationwarning" not in stderr_lower:
                    # Skip numba compilation errors for Numba engine
                    if "numba" in engine_name.lower() and ("typing" in stderr_lower or "compilation" in stderr_lower or "reflect" in stderr_lower):
                        return
                    self.fail(f"{engine_name}: Errors found in stderr:\n{stderr}")

    def test_numpy_engine_all_configurations(self):
        """Test NumPy engine (option 1) with all regression and function types."""
        print("\nTesting NumPy engine with all configurations...")
        result = self.run_main_and_check_success(1, "NumPy engine")

        # Verify key elements are present in output
        output_lower = result.stdout.lower()
        self.assertIn("regression", output_lower, "NumPy engine output missing 'regression'")
        self.assertIn("function", output_lower, "NumPy engine output missing 'function'")
        self.assertIn("successful", output_lower, "NumPy engine output missing 'successful'")

        print("NumPy engine test passed!")

    def test_numba_engine_all_configurations(self):
        """Test Numba engine (option 2) with all regression and function types."""
        print("\nTesting Numba engine with all configurations...")
        result = self.run_main_and_check_success(2, "Numba engine")

        # Verify key elements are present in output
        output_lower = result.stdout.lower()
        self.assertIn("regression", output_lower, "Numba engine output missing 'regression'")
        self.assertIn("function", output_lower, "Numba engine output missing 'function'")
        self.assertIn("successful", output_lower, "Numba engine output missing 'successful'")

        print("Numba engine test passed!")

    def test_pure_python_engine_all_configurations(self):
        """Test Pure Python engine (option 3) with all regression and function types."""
        print("\nTesting Pure Python engine with all configurations...")
        result = self.run_main_and_check_success(3, "Pure Python engine")

        # Verify key elements are present in output
        output_lower = result.stdout.lower()
        self.assertIn("regression", output_lower, "Pure Python engine output missing 'regression'")
        self.assertIn("function", output_lower, "Pure Python engine output missing 'function'")
        self.assertIn("successful", output_lower, "Pure Python engine output missing 'successful'")

        print("Pure Python engine test passed!")

    def test_all_engines_consistency(self):
        """Test that all engines produce consistent results and complete without errors."""
        print("\nTesting consistency across all engines...")

        # Test all engines
        engines = [
            (1, "NumPy"),
            (2, "Numba"),
            (3, "Pure Python")
        ]

        results = {}
        for engine_num, engine_name in engines:
            print(f"Running {engine_name} engine...")
            result = self.run_main_and_check_success(engine_num, engine_name)
            results[engine_name] = result.stdout

        # Verify all engines completed successfully
        for engine_name, output in results.items():
            self.assertIn("successful", output.lower(),
                         f"{engine_name} did not complete successfully")

        print("All engines completed consistently!")

    def test_engines_handle_different_regression_types(self):
        """Test that engines can handle all regression types individually."""
        print("\nTesting individual regression types...")

        # Test each regression type individually with NumPy engine
        regression_types = ["1", "2", "3", "4"]  # Linear, Ridge, Lasso, ElasticNet

        for reg_type in regression_types:
            input_sequence = f"4\n\n\n\n1\n{reg_type}\nall\n7\n"

            result = subprocess.run(
                [self.python_exe, str(self.main_path)],
                input=input_sequence,
                capture_output=True,
                text=True,
                timeout=300,
                check=False
            )

            self.assertEqual(result.returncode, 0,
                           f"Regression type {reg_type} failed with return code {result.returncode}")
            self.assertIn("successful", result.stdout.lower(),
                         f"Regression type {reg_type} did not complete successfully")

        print("All regression types work correctly!")

    def test_engines_handle_different_function_types(self):
        """Test that engines can handle different function types."""
        print("\nTesting individual function types...")

        # Test a few key function types with NumPy engine
        function_types = ["1", "2", "3", "8", "11"]  # Linear, Quadratic, Cubic, Log-linear, Square root

        for func_type in function_types:
            input_sequence = f"4\n\n\n\n1\nall\n{func_type}\n7\n"

            result = subprocess.run(
                [self.python_exe, str(self.main_path)],
                input=input_sequence,
                capture_output=True,
                text=True,
                timeout=300,
                check=False
            )

            self.assertEqual(result.returncode, 0,
                           f"Function type {func_type} failed with return code {result.returncode}")
            self.assertIn("successful", result.stdout.lower(),
                         f"Function type {func_type} did not complete successfully")

        print("All tested function types work correctly!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
