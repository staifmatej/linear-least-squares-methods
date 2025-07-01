"""Linear least squares approaches package."""

# Import all available engines
from . import least_squares_numpy
from . import least_squares_numba
from . import least_squares_pure

__all__ = [
    'least_squares_numpy',
    'least_squares_numba',
    'least_squares_pure'
]
