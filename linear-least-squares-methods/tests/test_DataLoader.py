"""Unit tests for DataLoader class."""

import warnings
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils.data_loader import DataLoader

warnings.filterwarnings('ignore')


# pylint: disable=too-many-public-methods
class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.data_loader = DataLoader()

    def test_init(self):
        """Test DataLoader initialization."""
        self.assertIsInstance(self.data_loader, DataLoader)

    @patch('builtins.input', side_effect=['1'])
    def test_show_menu_valid_choice(self, _):
        """Test show_menu with valid choice."""
        result = self.data_loader.show_menu()
        self.assertEqual(result, 1)

    @patch('builtins.input', side_effect=['0', '6', 'abc', '3'])
    def test_show_menu_invalid_then_valid_choice(self, _):
        """Test show_menu with invalid inputs then valid choice."""
        result = self.data_loader.show_menu()
        self.assertEqual(result, 3)

    @patch('builtins.input', side_effect=['1,2', '3,4', 'done'])
    def test_input_2d_points_valid_input(self, _):
        """Test input_2d_points with valid input."""
        x_data, y_data = self.data_loader.input_2d_points()
        np.testing.assert_array_equal(x_data, np.array([[1], [3]]))
        np.testing.assert_array_equal(y_data, np.array([2, 4]))

    @patch('builtins.input', side_effect=['1,2', 'done', '3,4', '5,6', 'done'])
    def test_input_2d_points_insufficient_points(self, _):
        """Test input_2d_points with insufficient points initially."""
        x_data, y_data = self.data_loader.input_2d_points()
        np.testing.assert_array_equal(x_data, np.array([[1], [3], [5]]))
        np.testing.assert_array_equal(y_data, np.array([2, 4, 6]))

    @patch('builtins.input', side_effect=['2', '1,2,3', '4,5,6', 'done'])
    def test_input_multidimensional_valid_input(self, _):
        """Test input_multidimensional with valid input."""
        x_data, y_data = self.data_loader.input_multidimensional()
        expected_x = np.array([[1, 2], [4, 5]])
        expected_y = np.array([3, 6])
        np.testing.assert_array_equal(x_data, expected_x)
        np.testing.assert_array_equal(y_data, expected_y)

    def test_find_csv_files(self):
        """Test _find_csv_files method."""
        csv_files = self.data_loader._find_csv_files()  # pylint: disable=protected-access
        self.assertIsInstance(csv_files, list)

    @patch('pandas.read_csv')
    def test_csv_files_read_data_valid_file(self, mock_read_csv):
        """Test _csv_files_read_data with valid CSV file."""
        mock_data = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'y': [2, 4, 6, 8]
        })
        mock_read_csv.return_value = mock_data

        with patch('builtins.input', side_effect=['y', 'x']):
            x_data, y_data = self.data_loader._csv_files_read_data('test.csv')  # pylint: disable=protected-access

        self.assertEqual(x_data.shape, (4, 1))
        self.assertEqual(y_data.shape, (4,))
        np.testing.assert_array_equal(y_data, np.array([2, 4, 6, 8]))

    @patch('pandas.read_csv')
    def test_csv_files_read_data_empty_file(self, mock_read_csv):
        """Test _csv_files_read_data with empty CSV file."""
        mock_read_csv.return_value = pd.DataFrame()

        with self.assertRaises(ValueError):
            self.data_loader._csv_files_read_data('empty.csv')  # pylint: disable=protected-access

    @patch('pandas.read_csv')
    def test_csv_files_read_data_insufficient_rows(self, mock_read_csv):
        """Test _csv_files_read_data with insufficient rows."""
        mock_data = pd.DataFrame({'x': [1], 'y': [2]})
        mock_read_csv.return_value = mock_data

        with self.assertRaises(ValueError):
            self.data_loader._csv_files_read_data('small.csv')  # pylint: disable=protected-access

    @patch('pandas.read_csv')
    def test_csv_files_read_data_insufficient_columns(self, mock_read_csv):
        """Test _csv_files_read_data with insufficient columns."""
        mock_data = pd.DataFrame({'x': [1, 2, 3]})
        mock_read_csv.return_value = mock_data

        with self.assertRaises(ValueError):
            self.data_loader._csv_files_read_data('narrow.csv')  # pylint: disable=protected-access

    def test_csv_files_not_found_single_file(self):
        """Test _csv_files_not_found with single file."""
        csv_files = ['test.csv']
        with patch('builtins.input', return_value='1'):
            filename, _ = self.data_loader._csv_files_not_found(None, csv_files)  # pylint: disable=protected-access
        self.assertEqual(filename, 'test.csv')

    def test_csv_files_not_found_custom_path(self):
        """Test _csv_files_not_found with custom path."""
        csv_files = ['test1.csv', 'test2.csv']
        with patch('builtins.input', side_effect=['0', 'custom.csv']):
            filename, _ = self.data_loader._csv_files_not_found(None, csv_files)  # pylint: disable=protected-access
        self.assertEqual(filename, 'custom.csv')

    @patch('builtins.input', side_effect=['50', '3', '0.1'])
    def test_generate_synthetic_data_default_values(self, _):
        """Test generate_synthetic_data with default-like values."""
        x_data, y_data = self.data_loader.generate_synthetic_data()
        self.assertEqual(x_data.shape, (50, 1))
        self.assertEqual(y_data.shape, (50,))
        self.assertTrue(np.all(np.isfinite(x_data)))
        self.assertTrue(np.all(np.isfinite(y_data)))

    @patch('builtins.input', side_effect=['1'])
    def test_use_example_dataset_house_prices(self, _):
        """Test use_example_dataset with house prices."""
        x_data, y_data = self.data_loader.use_example_dataset()
        self.assertEqual(x_data.shape[0], 9)
        self.assertEqual(y_data.shape[0], 9)
        self.assertTrue(np.all(x_data > 0))
        self.assertTrue(np.all(y_data > 0))

    @patch('builtins.input', side_effect=['2'])
    def test_use_example_dataset_temperature(self, _):
        """Test use_example_dataset with temperature data."""
        x_data, y_data = self.data_loader.use_example_dataset()
        self.assertEqual(x_data.shape[0], 25)
        self.assertEqual(y_data.shape[0], 25)

    @patch('builtins.input', side_effect=['3'])
    def test_use_example_dataset_quadratic(self, _):
        """Test use_example_dataset with quadratic function."""
        x_data, y_data = self.data_loader.use_example_dataset()
        self.assertEqual(x_data.shape[0], 20)
        self.assertEqual(y_data.shape[0], 20)

    @patch('builtins.input', side_effect=['4'])
    def test_use_example_dataset_sinusoidal(self, _):
        """Test use_example_dataset with sinusoidal function."""
        x_data, y_data = self.data_loader.use_example_dataset()
        self.assertEqual(x_data.shape[0], 100)
        self.assertEqual(y_data.shape[0], 100)

    @patch('builtins.input', side_effect=['0', '6', '2'])
    def test_use_example_dataset_invalid_then_valid(self, _):
        """Test use_example_dataset with invalid then valid choice."""
        x_data, _ = self.data_loader.use_example_dataset()
        self.assertEqual(x_data.shape[0], 25)

    @patch.object(DataLoader, 'input_2d_points')
    @patch('builtins.input', return_value='1')
    def test_get_data_choice_1(self, _, mock_method):
        """Test get_data with choice 1."""
        mock_method.return_value = (np.array([[1], [2]]), np.array([3, 4]))
        X, y = self.data_loader.get_data()
        mock_method.assert_called_once()
        np.testing.assert_array_equal(X, [[1], [2]])
        np.testing.assert_array_equal(y, [3, 4])

    @patch.object(DataLoader, 'input_multidimensional')
    @patch('builtins.input', return_value='2')
    def test_get_data_choice_2(self, _, mock_method):
        """Test get_data with choice 2."""
        mock_method.return_value = (np.array([[1, 2], [3, 4]]), np.array([5, 6]))
        X, y = self.data_loader.get_data()
        mock_method.assert_called_once()
        np.testing.assert_array_equal(X, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(y, [5, 6])

    @patch.object(DataLoader, 'load_from_csv')
    @patch('builtins.input', return_value='3')
    def test_get_data_choice_3(self, _, mock_method):
        """Test get_data with choice 3."""
        mock_method.return_value = (np.array([[1], [2]]), np.array([3, 4]))
        X, y = self.data_loader.get_data()
        mock_method.assert_called_once()
        np.testing.assert_array_equal(X, [[1], [2]])
        np.testing.assert_array_equal(y, [3, 4])

    @patch.object(DataLoader, 'generate_synthetic_data')
    @patch('builtins.input', return_value='4')
    def test_get_data_choice_4(self, _, mock_method):
        """Test get_data with choice 4."""
        mock_method.return_value = (np.array([[1], [2]]), np.array([3, 4]))
        X, y = self.data_loader.get_data()
        mock_method.assert_called_once()
        np.testing.assert_array_equal(X, [[1], [2]])
        np.testing.assert_array_equal(y, [3, 4])

    @patch.object(DataLoader, 'use_example_dataset')
    @patch('builtins.input', return_value='5')
    def test_get_data_choice_5(self, _, mock_method):
        """Test get_data with choice 5."""
        mock_method.return_value = (np.array([[1], [2]]), np.array([3, 4]))
        X, y = self.data_loader.get_data()
        mock_method.assert_called_once()
        np.testing.assert_array_equal(X, [[1], [2]])
        np.testing.assert_array_equal(y, [3, 4])


if __name__ == '__main__':
    unittest.main()
