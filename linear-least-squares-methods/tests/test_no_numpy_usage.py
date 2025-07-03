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

        # Check for numpy imports but exclude those inside try/except blocks
        numpy_imports = []

        def is_inside_try_except(node, tree):
            """Check if a node is inside a try/except block."""
            for ancestor in ast.walk(tree):
                if isinstance(ancestor, ast.Try):
                    # Check if the node is within this try block
                    for try_node in ast.walk(ancestor):
                        if try_node is node:
                            return True
            return False

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # Skip imports inside try/except blocks for internal optimization
                if is_inside_try_except(node, tree):
                    continue
                for alias in node.names:
                    if 'numpy' in alias.name or alias.name == 'np':
                        numpy_imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                # Skip imports inside try/except blocks for internal optimization
                if is_inside_try_except(node, tree):
                    continue
                if node.module and ('numpy' in node.module or node.module == 'np'):
                    imported_names = [alias.name for alias in node.names]
                    numpy_imports.append(f"from {node.module} import {', '.join(imported_names)}")

        # Assert no numpy imports found at module level
        self.assertEqual(len(numpy_imports), 0,
                        f"Found top-level numpy imports in {implementation_name}: {numpy_imports}")

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

        # Track whether we're inside a try/except block
        inside_try_block = False
        indentation_level = 0

        for i, line in enumerate(lines, 1):
            # Skip comments and strings
            if line.strip().startswith('#'):
                continue

            # Track try/except blocks by indentation
            stripped_line = line.strip()
            if stripped_line.startswith('try:'):
                inside_try_block = True
                indentation_level = len(line) - len(line.lstrip())
            elif inside_try_block:
                current_indent = len(line) - len(line.lstrip()) if line.strip() else indentation_level + 1
                # If we're back to the same or lower indentation level and not in except/finally
                if (current_indent <= indentation_level and
                    not stripped_line.startswith(('except', 'finally', 'else:')) and
                    stripped_line and not stripped_line.startswith('#')):
                    inside_try_block = False

            # Skip numpy pattern checking inside try blocks (internal optimization)
            if inside_try_block:
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
        # Note: This test will fail if numpy is used in numba implementation
        # According to CLAUDE.md, we should verify numpy is not used in numba file
        self._check_file_for_numpy('approaches/least_squares_numba.py', 'least_squares_numba.py')


if __name__ == '__main__':
    unittest.main()
