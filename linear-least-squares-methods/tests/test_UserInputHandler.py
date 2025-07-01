"""Unit tests for UserInputHandler class."""

import os
import sys
import unittest
from unittest.mock import patch
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# pylint: disable=wrong-import-position
from utils.user_input_handler import UserInputHandler


class TestUserInputHandler(unittest.TestCase):
    """Test UserInputHandler class functionality."""

    def setUp(self):
        """Set up test instance."""
        self.handler = UserInputHandler()

    def test_user_input_handler_creation(self):
        """Test that UserInputHandler can be created."""
        self.assertIsNotNone(self.handler)
        self.assertIsInstance(self.handler, UserInputHandler)

    @patch('builtins.input')
    def test_get_engine_choice_valid_numpy(self, mock_input):
        """Test engine choice selection for numpy."""
        mock_input.return_value = '1'
        choice = self.handler.get_engine_choice()
        self.assertEqual(choice, 1)

    @patch('builtins.input')
    def test_get_engine_choice_valid_numba(self, mock_input):
        """Test engine choice selection for numba."""
        mock_input.return_value = '2'
        choice = self.handler.get_engine_choice()
        self.assertEqual(choice, 2)

    @patch('builtins.input')
    def test_get_engine_choice_valid_pure(self, mock_input):
        """Test engine choice selection for pure python."""
        mock_input.return_value = '3'
        choice = self.handler.get_engine_choice()
        self.assertEqual(choice, 3)

    @patch('builtins.input')
    def test_get_engine_choice_invalid_then_valid(self, mock_input):
        """Test engine choice with invalid input followed by valid."""
        mock_input.side_effect = ['0', '5', '2']
        choice = self.handler.get_engine_choice()
        self.assertEqual(choice, 2)

    @patch('builtins.input')
    def test_get_engine_choice_non_numeric_then_valid(self, mock_input):
        """Test engine choice with non-numeric input followed by valid."""
        mock_input.side_effect = ['abc', '1']
        choice = self.handler.get_engine_choice()
        self.assertEqual(choice, 1)

    @patch('builtins.input')
    def test_get_regression_types_all(self, mock_input):
        """Test regression types selection with 'all'."""
        mock_input.return_value = 'all'
        types = self.handler.get_regression_types()
        self.assertEqual(types, [1, 2, 3, 4])

    @patch('builtins.input')
    def test_get_regression_types_single(self, mock_input):
        """Test regression types selection with single type."""
        mock_input.return_value = '1'
        types = self.handler.get_regression_types()
        self.assertEqual(types, [1])

    @patch('builtins.input')
    def test_get_regression_types_multiple(self, mock_input):
        """Test regression types selection with multiple types."""
        mock_input.return_value = '1,2,3'
        types = self.handler.get_regression_types()
        self.assertEqual(set(types), {1, 2, 3})

    @patch('builtins.input')
    def test_get_regression_types_with_spaces(self, mock_input):
        """Test regression types selection with spaces."""
        mock_input.return_value = '1, 2, 4'
        types = self.handler.get_regression_types()
        self.assertEqual(set(types), {1, 2, 4})

    @patch('builtins.input')
    def test_get_regression_types_invalid_then_valid(self, mock_input):
        """Test regression types with invalid input followed by valid."""
        mock_input.side_effect = ['1,5,6', '2,3']
        types = self.handler.get_regression_types()
        self.assertEqual(set(types), {2, 3})

    @patch('builtins.input')
    def test_get_function_types_all(self, mock_input):
        """Test function types selection with 'all'."""
        mock_input.return_value = 'all'
        types = self.handler.get_function_types()
        expected = list(range(1, 17))
        self.assertEqual(types, expected)

    @patch('builtins.input')
    def test_get_function_types_single(self, mock_input):
        """Test function types selection with single type."""
        mock_input.return_value = '1'
        types = self.handler.get_function_types()
        self.assertEqual(types, [1])

    @patch('builtins.input')
    def test_get_function_types_multiple(self, mock_input):
        """Test function types selection with multiple types."""
        mock_input.return_value = '1,2,7'
        types = self.handler.get_function_types()
        self.assertEqual(set(types), {1, 2, 7})

    @patch('builtins.input')
    def test_get_function_types_range(self, mock_input):
        """Test function types selection with comma-separated input."""
        mock_input.return_value = '1,2,3,4,5,6,7'
        types = self.handler.get_function_types()
        expected = [1, 2, 3, 4, 5, 6, 7]
        self.assertEqual(set(types), set(expected))

    @patch('builtins.input')
    def test_get_function_types_invalid_range_then_valid(self, mock_input):
        """Test function types with invalid input followed by valid."""
        mock_input.side_effect = ['20', '1,2']
        types = self.handler.get_function_types()
        self.assertEqual(set(types), {1, 2})

    @patch('builtins.input')
    def test_input_validation_regression_types(self, mock_input):
        """Test input validation for regression types."""
        mock_input.side_effect = ['abc', '0', '5', '1']
        types = self.handler.get_regression_types()
        self.assertEqual(types, [1])

    @patch('builtins.input')
    def test_input_validation_function_types(self, mock_input):
        """Test input validation for function types."""
        mock_input.side_effect = ['xyz', '0', '20', '2']
        types = self.handler.get_function_types()
        self.assertEqual(types, [2])

    def test_method_existence_validation(self):
        """Test that required methods exist."""
        self.assertTrue(hasattr(self.handler, 'get_engine_choice'))
        self.assertTrue(hasattr(self.handler, 'get_regression_types'))
        self.assertTrue(hasattr(self.handler, 'get_function_types'))


class TestUserInputHandlerEdgeCases(unittest.TestCase):
    """Test edge cases for UserInputHandler functionality."""

    def setUp(self):
        """Set up test instance."""
        self.handler = UserInputHandler()

    @patch('builtins.input')
    def test_empty_input_handling(self, mock_input):
        """Test handling of empty input."""
        mock_input.side_effect = ['', '1']
        choice = self.handler.get_engine_choice()
        self.assertEqual(choice, 1)

    @patch('builtins.input')
    def test_whitespace_input_handling(self, mock_input):
        """Test handling of whitespace input."""
        mock_input.side_effect = ['   ', '2']
        choice = self.handler.get_engine_choice()
        self.assertEqual(choice, 2)

    @patch('builtins.input')
    def test_case_insensitive_all(self, mock_input):
        """Test case insensitive 'all' input."""
        mock_input.return_value = 'ALL'
        types = self.handler.get_regression_types()
        self.assertEqual(types, [1, 2, 3, 4])

    @patch('builtins.input')
    def test_mixed_case_all(self, mock_input):
        """Test mixed case 'all' input."""
        mock_input.return_value = 'All'
        types = self.handler.get_regression_types()
        self.assertEqual(types, [1, 2, 3, 4])

    @patch('builtins.input')
    def test_duplicate_choices(self, mock_input):
        """Test handling of duplicate choices."""
        mock_input.return_value = '1,1,2,2,1'
        types = self.handler.get_regression_types()
        unique_types = list(set(types))
        self.assertEqual(set(unique_types), {1, 2})

    @patch('builtins.input')
    def test_negative_numbers(self, mock_input):
        """Test handling of negative numbers."""
        mock_input.side_effect = ['-1', '1']
        choice = self.handler.get_engine_choice()
        self.assertEqual(choice, 1)

    @patch('builtins.input')
    def test_float_input(self, mock_input):
        """Test handling of float input."""
        mock_input.side_effect = ['1.5', '2']
        choice = self.handler.get_engine_choice()
        self.assertEqual(choice, 2)


if __name__ == '__main__':
    unittest.main()
