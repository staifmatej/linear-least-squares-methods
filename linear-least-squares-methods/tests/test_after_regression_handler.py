"""Unit tests for after_regression_handler class."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import warnings

import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.after_regression_handler import (
    print_press_enter_to_continue,
    print_data_loaded,
    print_selected_specifications,
    print_selected_configurations,
    print_condition_numbers,
    print_coefficients
)


class TestAfterRegressionHandler(unittest.TestCase):
    """Test after regression handler functionality."""

    def setUp(self):
        """Set up test data."""
        self.mock_model = MagicMock()
        self.mock_model.coefficients = [1.0, 2.0, 0.5]
        self.mock_model.condition_number = 1.5

        self.sample_results = {
            (1, 1): {
                'model': self.mock_model,
                'mse': 0.123,
                'r2': 0.95,
                'mae': 0.087,
                'engine': 'numpy',
                'regression_type': 'Linear',
                'function_type': 1,
                'condition_number': 1.5
            },
            (2, 1): {
                'model': self.mock_model,
                'mse': 0.156,
                'r2': 0.92,
                'mae': 0.098,
                'engine': 'numpy',
                'regression_type': 'Ridge',
                'function_type': 1,
                'condition_number': 2.3
            }
        }

        self.sample_x = np.array([[1], [2], [3], [4], [5]])
        self.sample_y = np.array([2, 4, 6, 8, 10])

    def test_print_data_loaded_basic(self):
        """Test basic print_data_loaded functionality."""
        try:
            print_data_loaded(self.sample_x, self.sample_y)
            self.assertTrue(True)
        except Exception as exception:
            self.fail(f"print_data_loaded raised an exception: {exception}")

    def test_print_data_loaded_empty(self):
        """Test print_data_loaded with empty data."""
        try:
            print_data_loaded(np.array([]), np.array([]))
            self.assertTrue(True)
        except Exception as exception:
            self.fail(f"print_data_loaded with empty data raised an exception: {exception}")

    def test_print_selected_specifications_basic(self):
        """Test basic print_selected_specifications functionality."""
        try:
            print_selected_specifications(1, [1, 2], [1, 2, 3])
            self.assertTrue(True)
        except Exception as exception:
            self.fail(f"print_selected_specifications raised an exception: {exception}")

    def test_print_selected_specifications_single_values(self):
        """Test print_selected_specifications with single values."""
        try:
            print_selected_specifications(1, [1], [1])
            self.assertTrue(True)
        except Exception as exception:
            self.fail(f"print_selected_specifications single values raised an exception: {exception}")

    @patch('utils.after_regression_handler.print_press_enter_to_continue')
    def test_print_selected_configurations_basic(self, mock_continue):
        """Test basic print_selected_configurations functionality."""
        try:
            print_selected_configurations(1, [1, 2], [1, 2, 3])
            self.assertTrue(True)
        except Exception as exception:
            self.fail(f"print_selected_configurations raised an exception: {exception}")

    @patch('utils.after_regression_handler.print_press_enter_to_continue')
    def test_print_selected_configurations_empty_lists(self, mock_continue):
        """Test print_selected_configurations with empty lists."""
        try:
            print_selected_configurations(1, [], [])
            self.assertTrue(True)
        except Exception as exception:
            self.fail(f"print_selected_configurations empty lists raised an exception: {exception}")

    @patch('utils.after_regression_handler.print_press_enter_to_continue')
    def test_print_condition_numbers_basic(self, mock_continue):
        """Test basic print_condition_numbers functionality."""
        try:
            print_condition_numbers(self.sample_results, [1, 2], [1])
            self.assertTrue(True)
        except Exception as exception:
            self.fail(f"print_condition_numbers raised an exception: {exception}")

    @patch('utils.after_regression_handler.print_press_enter_to_continue')
    def test_print_condition_numbers_empty_results(self, mock_continue):
        """Test print_condition_numbers with empty results."""
        try:
            print_condition_numbers({}, [1], [1])
            self.assertTrue(True)
        except Exception as exception:
            self.fail(f"print_condition_numbers empty results raised an exception: {exception}")

    @patch('utils.after_regression_handler.print_press_enter_to_continue')
    def test_print_condition_numbers_none_values(self, mock_continue):
        """Test print_condition_numbers with None values."""
        none_results = {(1, 1): None}
        try:
            print_condition_numbers(none_results, [1], [1])
            self.assertTrue(True)
        except Exception as exception:
            self.fail(f"print_condition_numbers None values raised an exception: {exception}")

    def test_print_coefficients_function_exists(self):
        """Test that print_coefficients function exists."""
        self.assertTrue(callable(print_coefficients))

    def test_function_imports_work(self):
        """Test that all functions can be imported."""
        self.assertTrue(callable(print_condition_numbers))
        self.assertTrue(callable(print_coefficients))
        self.assertTrue(callable(print_data_loaded))
        self.assertTrue(callable(print_selected_specifications))
        self.assertTrue(callable(print_selected_configurations))


class TestAfterRegressionHandlerEdgeCases(unittest.TestCase):
    """Test edge cases for after regression handler functionality."""

    def test_functions_exist(self):
        """Test that all required functions exist."""
        functions = [
            print_press_enter_to_continue,
            print_data_loaded,
            print_selected_specifications,
            print_selected_configurations,
            print_condition_numbers,
            print_coefficients
        ]
        for func in functions:
            self.assertTrue(callable(func))


if __name__ == '__main__':
    unittest.main()