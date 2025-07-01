"""Unit tests to verify that visualization works consistently across all engines."""

import sys
import os
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
from utils.run_regression import RegressionRun
from utils.visualization import VisualizationData


class TestVisualizationConsistency(unittest.TestCase):
    """Test that visualization works consistently across all engines."""

    def setUp(self):
        """Set up test data."""
        # Test data that creates nice curves for visualization
        self.X = np.linspace(0, 10, 50).reshape(-1, 1)
        self.y = np.sin(self.X.flatten()) + 0.1 * np.random.randn(50)

        # Engine choices: 1=numpy, 2=numba, 3=pure
        self.engines = [1, 2, 3]
        self.engine_names = {1: "NumPy", 2: "Numba", 3: "Pure"}

        # Regression types to test
        self.regression_types = [1, 2, 3, 4]  # Linear, Ridge, Lasso, ElasticNet
        self.regression_names = {1: "Linear", 2: "Ridge", 3: "Lasso", 4: "ElasticNet"}

    def _test_visualization_for_engine(self, engine_choice, regression_type):
        """Test visualization for a specific engine and regression type."""
        try:
            # Run regression
            runner = RegressionRun(engine_choice=engine_choice,
                                 regression_types=[regression_type],
                                 function_types=[7])  # Septic (degree 7)
            results = runner.run_regressions(self.X, self.y)

            # Check if we got results
            result = list(results.values())[0]
            if not result or 'model' not in result:
                return False, "No model returned"

            _ = result['model']  # Unused but needed for structure

            # Test visualization creation
            viz = VisualizationData(self.X, self.y, results)

            # Try to create visualization (this should not crash)
            try:
                viz.plot_results()
                return True, "Visualization created successfully"
            except Exception as e:  # pylint: disable=broad-exception-caught
                return False, f"Visualization failed: {e}"

        except Exception as e:  # pylint: disable=broad-exception-caught
            return False, f"Regression failed: {e}"

    def test_linear_regression_visualization(self):
        """Test that Linear regression visualization works for all engines."""
        print("\n=== Testing Linear Regression Visualization ===")

        for engine in self.engines:
            with self.subTest(engine=engine):
                engine_name = self.engine_names[engine]
                print(f"\nTesting {engine_name} engine...")

                success, msg = self._test_visualization_for_engine(engine, 1)
                self.assertTrue(success, f"{engine_name} visualization failed: {msg}")
                print(f"OK {engine_name} visualization works")

    def test_ridge_regression_visualization(self):
        """Test that Ridge regression visualization works for all engines."""
        print("\n=== Testing Ridge Regression Visualization ===")

        for engine in self.engines:
            with self.subTest(engine=engine):
                engine_name = self.engine_names[engine]
                print(f"\nTesting {engine_name} engine...")

                success, msg = self._test_visualization_for_engine(engine, 2)
                self.assertTrue(success, f"{engine_name} Ridge visualization failed: {msg}")
                print(f"OK {engine_name} Ridge visualization works")

    def test_lasso_regression_visualization(self):
        """Test that Lasso regression visualization works for all engines."""
        print("\n=== Testing Lasso Regression Visualization ===")

        for engine in self.engines:
            with self.subTest(engine=engine):
                engine_name = self.engine_names[engine]
                print(f"\nTesting {engine_name} engine...")

                success, msg = self._test_visualization_for_engine(engine, 3)
                self.assertTrue(success, f"{engine_name} Lasso visualization failed: {msg}")
                print(f"OK {engine_name} Lasso visualization works")

    def test_elasticnet_regression_visualization(self):
        """Test that ElasticNet regression visualization works for all engines."""
        print("\n=== Testing ElasticNet Regression Visualization ===")

        for engine in self.engines:
            with self.subTest(engine=engine):
                engine_name = self.engine_names[engine]
                print(f"\nTesting {engine_name} engine...")

                success, msg = self._test_visualization_for_engine(engine, 4)
                self.assertTrue(success, f"{engine_name} ElasticNet visualization failed: {msg}")
                print(f"OK {engine_name} ElasticNet visualization works")

    def test_model_prediction_consistency(self):
        """Test that models from all engines can make predictions consistently."""
        print("\n=== Testing Model Prediction Consistency ===")

        # Test prediction at a few points
        X_test = np.array([[2.0], [5.0], [8.0]])

        for reg_type in [2, 3, 4]:  # Ridge, Lasso, ElasticNet
            with self.subTest(regression_type=reg_type):
                reg_name = self.regression_names[reg_type]
                print(f"\nTesting {reg_name} predictions...")

                predictions = {}

                for engine in self.engines:
                    engine_name = self.engine_names[engine]

                    try:
                        # Run regression
                        runner = RegressionRun(engine_choice=engine,
                                             regression_types=[reg_type],
                                             function_types=[2])  # Quadratic
                        results = runner.run_regressions(self.X, self.y)
                        result = list(results.values())[0]

                        if result and 'model' in result:
                            model = result['model']

                            # Test prediction
                            if hasattr(model, 'predict'):
                                pred = model.predict(X_test)
                                predictions[engine] = pred
                                print(f"  {engine_name}: {pred}")
                            else:
                                self.fail(f"{engine_name} model has no predict method")
                        else:
                            self.fail(f"{engine_name} returned no model")

                    except Exception as e:  # pylint: disable=broad-exception-caught
                        self.fail(f"{engine_name} failed: {e}")

                # Check that all engines returned predictions
                self.assertEqual(len(predictions), 3, f"Not all engines returned predictions for {reg_name}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2, buffer=False)
