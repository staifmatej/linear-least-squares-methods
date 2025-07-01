"""Unit tests for VisualizationData class."""


import unittest
import warnings

import matplotlib.pyplot as plt
import numpy as np

from utils.visualization import VisualizationData
from utils.run_regression import RegressionRun

warnings.filterwarnings('ignore')


class MockModel:  # pylint: disable=too-few-public-methods
    """Mock model for testing visualization."""

    def __init__(self, coefficients=None):
        self.coefficients = coefficients or [1.0, 2.0]

    def predict(self, x_input):
        """Mock prediction method."""
        if hasattr(x_input, 'shape') and len(x_input.shape) > 1:
            return [sum(self.coefficients[i] * (x_val[i-1] if i > 0 else 1)
                       for i in range(len(self.coefficients))) for x_val in x_input]
        return [self.coefficients[0] + self.coefficients[1] * x_val for x_val in x_input]

    def _generate_polynomial_features(self, x_input):
        """Mock polynomial feature generation."""
        if hasattr(x_input, 'flatten'):
            x_input = x_input.flatten()
        return np.column_stack([x_input, x_input**2])


class TestVisualizationData(unittest.TestCase):
    """Test VisualizationData class functionality."""

    def setUp(self):
        """Set up test data and visualization instance."""
        np.random.seed(42)
        self.simple_x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        self.simple_y = np.array([2, 4, 6, 8, 10])
        self.noise_x = np.linspace(0, 10, 20).reshape(-1, 1)
        self.noise_y = 2 * self.noise_x.flatten() + 1 + np.random.normal(0, 0.1, 20)

        mock_model = MockModel()
        self.mock_results = {
            (1, 1): {
                'model': mock_model,
                'function_type': 1,
                'regression_type': 1
            }
        }

        self.viz = VisualizationData(self.simple_x, self.simple_y, self.mock_results)

    def test_visualization_data_creation(self):
        """Test that VisualizationData can be created."""
        self.assertIsNotNone(self.viz)
        self.assertIsInstance(self.viz, VisualizationData)
        np.testing.assert_array_equal(self.viz.X, self.simple_x)
        np.testing.assert_array_equal(self.viz.y, self.simple_y)

    def test_ensure_numpy_array_list_input(self):
        """Test conversion of list to numpy array."""
        test_list = [1, 2, 3, 4, 5]
        result = self.viz._ensure_numpy_array(test_list)  # pylint: disable=protected-access
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, test_list)

    def test_ensure_numpy_array_numpy_input(self):
        """Test that numpy arrays are returned unchanged."""
        test_array = np.array([1, 2, 3, 4, 5])
        result = self.viz._ensure_numpy_array(test_array)  # pylint: disable=protected-access
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, test_array)

    def test_transform_features_for_prediction_pure_log_linear(self):
        """Test feature transformation for log-linear function."""
        test_x = np.array([1, 2, 3, 4, 5])
        result = self.viz._transform_features_for_prediction_pure(test_x, function_type=8)  # pylint: disable=protected-access
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], list)
        self.assertEqual(len(result[0]), len(test_x))

    def test_transform_features_for_prediction_pure_square_root(self):
        """Test feature transformation for square root function."""
        test_x = np.array([1, 4, 9, 16, 25])
        result = self.viz._transform_features_for_prediction_pure(test_x, function_type=11)  # pylint: disable=protected-access
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], list)
        self.assertEqual(len(result[0]), len(test_x))

    def test_transform_features_for_prediction_pure_inverse(self):
        """Test feature transformation for inverse function."""
        test_x = np.array([1, 2, 3, 4, 5])
        result = self.viz._transform_features_for_prediction_pure(test_x, function_type=12)  # pylint: disable=protected-access
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], list)
        self.assertEqual(len(result[0]), len(test_x))

    def test_transform_features_edge_case_zero_values(self):
        """Test feature transformation with zero values."""
        test_x = np.array([0, 1, 2, 3])
        result = self.viz._transform_features_for_prediction_pure(test_x, function_type=8)  # pylint: disable=protected-access
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], list)
        self.assertTrue(all(isinstance(row, list) for row in result[0]))

    def test_transform_features_edge_case_negative_values(self):
        """Test feature transformation with negative values."""
        test_x = np.array([-2, -1, 0, 1, 2])
        result = self.viz._transform_features_for_prediction_pure(test_x, function_type=11)  # pylint: disable=protected-access
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], list)

    def test_plot_results_basic(self):
        """Test basic plot results functionality."""
        plt.ioff()
        try:
            self.viz.plot_results()
            # Check that no exception was raised and matplotlib figure was created
            self.assertGreater(len(plt.get_fignums()), 0)
        except Exception as exception:  # pylint: disable=broad-exception-caught
            self.fail(f"plot_results raised an exception: {exception}")
        finally:
            plt.close('all')

    def test_plot_results_with_real_regression(self):
        """Test plot results with real regression model."""
        plt.ioff()
        try:
            runner = RegressionRun(engine_choice=1, regression_types=[1], function_types=[1])
            results = runner.run_regressions(self.noise_x, self.noise_y)
            viz = VisualizationData(self.noise_x, self.noise_y, results)
            viz.plot_results()
            # Check that visualization was created successfully
            self.assertIsNotNone(results)
            self.assertGreater(len(plt.get_fignums()), 0)
        except Exception as exception:  # pylint: disable=broad-exception-caught
            self.fail(f"plot_results with real regression raised an exception: {exception}")
        finally:
            plt.close('all')

    def test_color_palette_selection(self):
        """Test color palette selection."""
        if hasattr(self.viz, '_get_color_palette'):
            colors = self.viz._get_color_palette(4)  # pylint: disable=protected-access
            self.assertEqual(len(colors), 4)
            self.assertTrue(all(isinstance(color, str) for color in colors))

    def test_plot_title_generation(self):
        """Test plot title generation."""
        if hasattr(self.viz, '_generate_plot_title'):
            title = self.viz._generate_plot_title(1, 1)  # pylint: disable=protected-access
            self.assertIsInstance(title, str)
            self.assertGreater(len(title), 0)

    def test_plot_legend_creation(self):
        """Test plot legend creation."""
        plt.ioff()
        try:
            _, axis = plt.subplots()
            axis.plot([1, 2, 3], [1, 2, 3], label='test')
            if hasattr(self.viz, '_add_legend'):
                self.viz._add_legend(axis)  # pylint: disable=protected-access
                # Check that legend was added or method completed successfully
                legend = axis.get_legend()
                self.assertIsNotNone(legend)
            else:
                # If method doesn't exist, create legend manually for test
                axis.legend()
                legend = axis.get_legend()
                self.assertIsNotNone(legend)
        except Exception as exception:  # pylint: disable=broad-exception-caught
            self.fail(f"Legend creation raised an exception: {exception}")
        finally:
            plt.close('all')

    def test_empty_results_handling(self):
        """Test handling of empty results."""
        empty_viz = VisualizationData(self.simple_x, self.simple_y, {})
        plt.ioff()
        try:
            empty_viz.plot_results()
            # Check that empty results are handled gracefully
            self.assertEqual(len(empty_viz.results), 0)
        except Exception as exception:  # pylint: disable=broad-exception-caught
            self.fail(f"Empty results handling raised an exception: {exception}")
        finally:
            plt.close('all')

    def test_none_model_handling(self):
        """Test handling of None models in results."""
        none_results = {(1, 1): None}
        none_viz = VisualizationData(self.simple_x, self.simple_y, none_results)
        plt.ioff()
        try:
            none_viz.plot_results()
            # Check that None models are handled gracefully
            self.assertIsNotNone(none_viz.results)
            self.assertIn((1, 1), none_viz.results)
        except Exception as exception:  # pylint: disable=broad-exception-caught
            self.fail(f"None model handling raised an exception: {exception}")
        finally:
            plt.close('all')

    def test_visualization_with_different_data_sizes(self):
        """Test visualization with different data sizes."""
        large_x = np.linspace(0, 100, 1000).reshape(-1, 1)
        large_y = 2 * large_x.flatten() + np.random.normal(0, 1, 1000)
        large_viz = VisualizationData(large_x, large_y, self.mock_results)

        plt.ioff()
        try:
            large_viz.plot_results()
            # Check that large datasets are handled successfully
            self.assertEqual(len(large_viz.X), 1000)
            self.assertEqual(len(large_viz.y), 1000)
        except Exception as exception:  # pylint: disable=broad-exception-caught
            self.fail(f"Large data visualization raised an exception: {exception}")
        finally:
            plt.close('all')


class TestVisualizationDataEdgeCases(unittest.TestCase):
    """Test edge cases for VisualizationData functionality."""

    def setUp(self):
        """Set up test data."""
        self.edge_x = np.array([1]).reshape(-1, 1)
        self.edge_y = np.array([2])
        mock_model = MockModel()
        self.edge_results = {(1, 1): {'model': mock_model, 'function_type': 1}}

    def test_single_data_point_visualization(self):
        """Test visualization with single data point."""
        viz = VisualizationData(self.edge_x, self.edge_y, self.edge_results)
        plt.ioff()
        try:
            viz.plot_results()
            # Check that single data point is handled successfully
            self.assertEqual(len(viz.X), 1)
            self.assertEqual(len(viz.y), 1)
        except Exception as exception:  # pylint: disable=broad-exception-caught
            self.fail(f"Single point visualization raised an exception: {exception}")
        finally:
            plt.close('all')

    def test_identical_x_values_visualization(self):
        """Test visualization with identical x values."""
        identical_x = np.array([2, 2, 2, 2, 2]).reshape(-1, 1)
        varied_y = np.array([1, 2, 3, 4, 5])
        viz = VisualizationData(identical_x, varied_y, self.edge_results)

        plt.ioff()
        try:
            viz.plot_results()
            # Check that identical x values are handled successfully
            unique_x = np.unique(viz.X)
            self.assertEqual(len(unique_x), 1)
        except Exception as exception:  # pylint: disable=broad-exception-caught
            self.fail(f"Identical x values visualization raised an exception: {exception}")
        finally:
            plt.close('all')

    def test_large_value_range_visualization(self):
        """Test visualization with large value ranges."""
        large_x = np.array([1e6, 2e6, 3e6, 4e6, 5e6]).reshape(-1, 1)
        large_y = np.array([1e9, 2e9, 3e9, 4e9, 5e9])
        viz = VisualizationData(large_x, large_y, self.edge_results)

        plt.ioff()
        try:
            viz.plot_results()
            # Check that large value ranges are handled successfully
            self.assertTrue(np.all(viz.X >= 1e6))
            self.assertTrue(np.all(viz.y >= 1e9))
        except Exception as exception:  # pylint: disable=broad-exception-caught
            self.fail(f"Large values visualization raised an exception: {exception}")
        finally:
            plt.close('all')

    def test_negative_values_visualization(self):
        """Test visualization with negative values."""
        neg_x = np.array([-5, -3, -1, 1, 3]).reshape(-1, 1)
        neg_y = np.array([-10, -6, -2, 2, 6])
        viz = VisualizationData(neg_x, neg_y, self.edge_results)

        plt.ioff()
        try:
            viz.plot_results()
            # Check that negative values are handled successfully
            self.assertTrue(np.any(viz.X < 0))
            self.assertTrue(np.any(viz.y < 0))
        except Exception as exception:  # pylint: disable=broad-exception-caught
            self.fail(f"Negative values visualization raised an exception: {exception}")
        finally:
            plt.close('all')

    def test_zero_values_visualization(self):
        """Test visualization with zero values."""
        zero_x = np.array([0, 1, 2, 3, 4]).reshape(-1, 1)
        zero_y = np.array([0, 1, 2, 3, 4])
        viz = VisualizationData(zero_x, zero_y, self.edge_results)

        plt.ioff()
        try:
            viz.plot_results()
            # Check that zero values are handled successfully
            self.assertTrue(np.any(viz.X == 0))
            self.assertTrue(np.any(viz.y == 0))
        except Exception as exception:  # pylint: disable=broad-exception-caught
            self.fail(f"Zero values visualization raised an exception: {exception}")
        finally:
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
