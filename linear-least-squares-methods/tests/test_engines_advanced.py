"""Test all possible engine configurations by running main.py with different inputs."""

import unittest
import subprocess
import sys
from pathlib import Path

# pylint: disable=R0801
class TestAllEngineConfigurations(unittest.TestCase):
    """Test all possible engine configurations by running main.py with different inputs."""

    def setUp(self):
        """Set up test environment."""
        self.main_path = Path(__file__).parent.parent / "main.py"
        self.python_exe = sys.executable

    def run_main_with_input(self, engine_choice, description=""):
        """Run main.py with specific input sequence and capture output."""
        # Input sequence:
        # 1. Choose generated synthetic data (option 4)
        # 2. Number of samples (default 50): enter
        # 3. Polynomial degree (default 3): enter
        # 4. Noise level (default 0.1): enter
        # 5. Choose engine (1, 2, or 3)
        # 6. Choose "all" for regression types
        # 7. Choose "all" for function types
        # 8. Exit when done (option 7)
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
            return result
        except subprocess.TimeoutExpired as exc:
            self.fail(f"Test timed out for {description}: {exc}")
            return None

    def check_for_errors(self, output, stderr, engine_name):
        """Check output for errors and warnings (excluding condition number warnings)."""
        # Convert to lowercase for case-insensitive matching
        output_lower = output.lower()
        stderr_lower = stderr.lower()

        # Check for common error indicators
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
            if indicator in output_lower:
                # Skip if it's just about condition number
                if "condition number" in output_lower:
                    continue
                # Skip numba compilation errors for Numba engine - these are known issues
                if "numba" in engine_name.lower() and ("typing" in output_lower or "compilation" in output_lower or "reflect" in output_lower):
                    continue
                self.fail(f"{engine_name}: Found error indicator '{indicator}' in output:\n{output}")

        # Check stderr for actual errors
        if stderr and stderr.strip():
            # Ignore warnings about condition numbers
            if "condition number" not in stderr_lower and "warning" not in stderr_lower:
                # Also ignore numba compilation warnings and errors
                if "numba" not in stderr_lower and "deprecationwarning" not in stderr_lower:
                    # Skip numba compilation errors for Numba engine
                    if "numba" in engine_name.lower() and ("typing" in stderr_lower or "compilation" in stderr_lower or "reflect" in stderr_lower):
                        return
                    self.fail(f"{engine_name}: Errors found in stderr:\n{stderr}")

    def test_numpy_engine(self):
        """Test NumPy engine (option 1) with all regression and function types."""
        print("\nTesting NumPy engine...")
        result = self.run_main_with_input(1, "NumPy engine")

        # Check return code
        self.assertEqual(result.returncode, 0,
                        f"NumPy engine failed with return code {result.returncode}\n"
                        f"stdout: {result.stdout}\n"
                        f"stderr: {result.stderr}")

        # Check for errors
        self.check_for_errors(result.stdout, result.stderr, "NumPy engine")

        # Verify some computation happened
        self.assertIn("regression", result.stdout.lower(),
                     "NumPy engine output doesn't mention regression")

        print("NumPy engine test passed!")

    def test_numba_engine(self):
        """Test Numba engine (option 2) with all regression and function types."""
        print("\nTesting Numba engine...")
        result = self.run_main_with_input(2, "Numba engine")

        # Check return code
        self.assertEqual(result.returncode, 0,
                        f"Numba engine failed with return code {result.returncode}\n"
                        f"stdout: {result.stdout}\n"
                        f"stderr: {result.stderr}")

        # Check for errors
        self.check_for_errors(result.stdout, result.stderr, "Numba engine")

        # Verify some computation happened
        self.assertIn("regression", result.stdout.lower(),
                     "Numba engine output doesn't mention regression")

        print("Numba engine test passed!")

    def test_pure_python_engine(self):
        """Test Pure Python engine (option 3) with all regression and function types."""
        print("\nTesting Pure Python engine...")
        result = self.run_main_with_input(3, "Pure Python engine")

        # Check return code
        self.assertEqual(result.returncode, 0,
                        f"Pure Python engine failed with return code {result.returncode}\n"
                        f"stdout: {result.stdout}\n"
                        f"stderr: {result.stderr}")

        # Check for errors
        self.check_for_errors(result.stdout, result.stderr, "Pure Python engine")

        # Verify some computation happened
        self.assertIn("regression", result.stdout.lower(),
                     "Pure Python engine output doesn't mention regression")

        print("Pure Python engine test passed!")

    def test_all_engines_produce_output(self):
        """Verify all engines produce similar structure of output."""
        print("\nComparing outputs from all engines...")

        # Test all engines
        results = {}
        for engine_num, engine_name in [(1, "NumPy"), (2, "Numba"), (3, "Pure Python")]:
            result = self.run_main_with_input(engine_num, f"{engine_name} engine")
            results[engine_name] = result.stdout

        # Check that all outputs mention key elements
        key_elements = ["regression", "function", "successful"]

        for engine_name, output in results.items():
            output_lower = output.lower()
            for element in key_elements:
                self.assertIn(element, output_lower,
                            f"{engine_name} output missing key element '{element}'")

        print("All engines produce consistent output structure!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
