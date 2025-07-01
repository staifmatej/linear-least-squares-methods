"""Unit tests to verify that all engines return consistent coefficients for the same data."""

import unittest
import numpy as np

from utils.run_regression import RegressionRun


# pylint: disable=too-many-instance-attributes
class TestEngineConsistency(unittest.TestCase):
    """Test that all engines return consistent coefficients for identical inputs."""

    def setUp(self):
        """Set up test data."""
        # Simple linear test data (5 points - allows up to degree 4)
        self.X_simple = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        self.y_simple = np.array([2.1, 3.9, 6.2, 7.8, 10.1])

        # More complex test data (6 points - allows up to degree 5)
        self.X_complex = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]).reshape(-1, 1)
        self.y_complex = np.array([1.2, 2.8, 5.1, 8.9, 14.2, 21.5])

        # Tolerance for floating point comparisons
        self.tolerance = 1e-3

        # Engine choices: 1=numpy, 2=numba, 3=pure
        self.engines = [1, 2, 3]
        self.engine_names = {1: "NumPy", 2: "Numba", 3: "Pure"}

        # Function types that work reliably across all engines
        self.polynomial_functions = [1, 2, 3, 4]  # Valid for 5 data points
        self.special_functions = [8, 11, 12]  # Log-Linear, Square Root, Inverse

    def _compare_coefficients(self, coeffs1, coeffs2, tolerance=None):
        """Compare two coefficient arrays with tolerance."""
        if tolerance is None:
            tolerance = self.tolerance

        # Convert to numpy arrays for comparison
        c1 = np.array(coeffs1, dtype=float)
        c2 = np.array(coeffs2, dtype=float)

        # Check if shapes match
        if c1.shape != c2.shape:
            return False, f"Shape mismatch: {c1.shape} vs {c2.shape}"

        # Check if values are close
        if not np.allclose(c1, c2, atol=tolerance, rtol=tolerance):
            diff = np.abs(c1 - c2)
            max_diff = np.max(diff)
            return False, f"Max difference: {max_diff:.6f}, tolerance: {tolerance}"

        return True, "Coefficients match"

    def _run_single_test(self, X, y, reg_type, func_type):
        """Run regression with all engines and compare results."""
        results = {}

        # Run regression with each engine
        for engine in self.engines:
            try:
                runner = RegressionRun(engine_choice=engine,
                                     regression_types=[reg_type],
                                     function_types=[func_type])
                engine_results = runner.run_regressions(X, y)
                result = engine_results.get((reg_type, func_type))

                if result and 'coefficients' in result:
                    results[engine] = result['coefficients']
                else:
                    results[engine] = None

            except (ValueError, RuntimeError, TypeError) as exc:
                results[engine] = None
                print(f"Engine {self.engine_names[engine]} failed: {exc}")

        return results

    def test_polynomial_linear_regression(self):
        """Test Linear regression with polynomial functions - CRITICAL."""
        print("\n=== Testing Linear Regression with Polynomial Functions ===")

        for func_type in self.polynomial_functions:
            with self.subTest(function_type=func_type):
                print(f"\nTesting polynomial degree {func_type}...")

                results = self._run_single_test(self.X_simple, self.y_simple, 1, func_type)

                # Use NumPy as reference
                numpy_coeffs = results.get(1)
                self.assertIsNotNone(numpy_coeffs, "NumPy engine should return coefficients")

                # Compare other engines to NumPy
                for engine in [2, 3]:  # Numba, Pure
                    engine_coeffs = results.get(engine)
                    if engine_coeffs is not None:
                        is_close, msg = self._compare_coefficients(numpy_coeffs, engine_coeffs)
                        self.assertTrue(is_close,
                                      f"Linear regression degree {func_type}: {self.engine_names[engine]} vs NumPy - {msg}\n"
                                      f"NumPy: {numpy_coeffs}\n{self.engine_names[engine]}: {engine_coeffs}")
                        print(f"OK {self.engine_names[engine]} matches NumPy")
                    else:
                        self.fail(f"{self.engine_names[engine]} failed for polynomial degree {func_type}")

    def test_special_functions_linear_regression(self):
        """Test Linear regression with special functions - CRITICAL."""
        print("\n=== Testing Linear Regression with Special Functions ===")

        # Use positive data for special functions
        X_special = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        y_special = np.array([2.1, 3.9, 6.2, 7.8, 10.1])

        function_names = {8: "Log-Linear", 11: "Square Root", 12: "Inverse"}

        for func_type in self.special_functions:
            with self.subTest(function_type=func_type):
                print(f"\nTesting {function_names[func_type]} function...")

                results = self._run_single_test(X_special, y_special, 1, func_type)

                # Use NumPy as reference
                numpy_coeffs = results.get(1)
                self.assertIsNotNone(numpy_coeffs, f"NumPy engine should return coefficients for {function_names[func_type]}")

                # Compare other engines to NumPy
                for engine in [2, 3]:  # Numba, Pure
                    engine_coeffs = results.get(engine)
                    if engine_coeffs is not None:
                        is_close, msg = self._compare_coefficients(numpy_coeffs, engine_coeffs)
                        self.assertTrue(is_close,
                                      f"{function_names[func_type]} regression: {self.engine_names[engine]} vs NumPy - {msg}\n"
                                      f"NumPy: {numpy_coeffs}\n{self.engine_names[engine]}: {engine_coeffs}")
                        print(f"OK {self.engine_names[engine]} matches NumPy")
                    else:
                        print(f"Warning: {self.engine_names[engine]} failed for {function_names[func_type]}")

    def test_underdetermined_systems_rejection(self):
        """Test that all engines properly reject underdetermined systems."""
        print("\n=== Testing Underdetermined Systems Rejection ===")

        # Test with 5 data points and degree 5+ (should fail)
        for degree in [5, 6, 7]:
            with self.subTest(degree=degree):
                print(f"\nTesting degree {degree} with 5 data points (should fail)...")

                results = self._run_single_test(self.X_simple, self.y_simple, 1, degree)

                # All engines should return None (failure)
                for engine in self.engines:
                    engine_coeffs = results.get(engine)
                    self.assertIsNone(engine_coeffs,
                                    f"{self.engine_names[engine]} should reject underdetermined system "
                                    f"(degree {degree} with 5 data points)")
                    print(f"OK {self.engine_names[engine]} properly rejects underdetermined system")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2, buffer=False)
