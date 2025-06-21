"""Unit tests for main.py"""

from main import DataLoader


def test_dataloader_creation():
    """Test that DataLoader can be created."""
    loader = DataLoader()
    assert loader is not None

