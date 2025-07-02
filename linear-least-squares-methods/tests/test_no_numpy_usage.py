"""Test to verify no numpy usage in pure and numba implementations."""

import ast
import unittest


class TestNoNumpyUsage(unittest.TestCase):
    """Test that least_squares_pure.py and least_squares_numba.py don't use numpy."""

    def _check_file_for_numpy(self, filepath, implementation_name):  # pylint: disable=too-many-locals,too-many-branches
        """Helper method to check a file for numpy usage."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse the AST
        tree = ast.parse(content)

        # Check for numpy imports
        numpy_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if 'numpy' in alias.name or alias.name == 'np':
                        numpy_imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and ('numpy' in node.module or node.module == 'np'):
                    imported_names = [alias.name for alias in node.names]
                    numpy_imports.append(f"from {node.module} import {', '.join(imported_names)}")

        # Assert no numpy imports found
        self.assertEqual(len(numpy_imports), 0,
                        f"Found numpy imports in {implementation_name}: {numpy_imports}")

        # Check for common numpy usage patterns
        numpy_patterns = [
            'np.',
            'numpy.',
            'ndarray',
            'asarray',
            'arange',
            'linspace',
            'zeros_like',
            'ones_like',
        ]

        found_patterns = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Skip comments and strings
            if line.strip().startswith('#'):
                continue

            # Remove string literals to avoid false positives
            try:
                # Simple approach: remove content between quotes
                line_without_strings = line
                for quote in ["'''", '"""', "'", '"']:
                    parts = line_without_strings.split(quote)
                    if len(parts) > 1:
                        # Keep only parts outside quotes
                        line_without_strings = parts[0] + ' '.join(parts[2::2])

                for pattern in numpy_patterns:
                    if pattern in line_without_strings:
                        found_patterns.append(f"Line {i}: {line.strip()}")
            except Exception:  # pylint: disable=broad-exception-caught
                # If parsing fails, check the original line
                for pattern in numpy_patterns:
                    if pattern in line:
                        found_patterns.append(f"Line {i}: {line.strip()}")

        # Assert no numpy usage found
        self.assertEqual(len(found_patterns), 0,
                        f"Found numpy usage in {implementation_name}:\n" + '\n'.join(found_patterns))

    def test_no_numpy_in_pure_implementation(self):
        """Check that numpy is not used in least_squares_pure.py"""
        self._check_file_for_numpy('approaches/least_squares_pure.py', 'least_squares_pure.py')

    def test_no_numpy_in_numba_implementation(self):
        """Check that numpy is not used in least_squares_numba.py"""
        self._check_file_for_numpy('approaches/least_squares_numba.py', 'least_squares_numba.py')


if __name__ == '__main__':
    unittest.main()
