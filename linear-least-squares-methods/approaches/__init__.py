"""Linear least squares approaches package."""

# Import all available engines
from . import least_squares_numpy
from . import least_squares_numba  
from . import least_squares_pure

# Try to import C++ wrapper - this will handle the import gracefully
try:
    from . import least_squares_cpp_wrapper
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

__all__ = [
    'least_squares_numpy',
    'least_squares_numba', 
    'least_squares_pure',
    'least_squares_cpp_wrapper',
    'CPP_AVAILABLE'
]