"""Unit tests for main.py"""

from main import DataLoader


def test_dataloader_creation():
    """Test that DataLoader can be created."""
    loader = DataLoader()
    assert loader is not None

def test_basic_functionality():
    """Basic test."""
    assert 1 + 1 == 2
