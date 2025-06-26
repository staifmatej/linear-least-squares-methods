from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import subprocess
import sys
import os

def get_cmake_dir():
    """Get the directory containing CMakeLists.txt"""
    return os.path.dirname(os.path.abspath(__file__))

def build_cpp_module():
    """Build the C++ module using CMake"""
    cmake_dir = get_cmake_dir()
    build_dir = os.path.join(cmake_dir, "build")
    
    # Create build directory
    os.makedirs(build_dir, exist_ok=True)
    
    # Run CMake
    try:
        subprocess.run([
            "cmake", 
            "-S", cmake_dir,
            "-B", build_dir,
            "-DCMAKE_BUILD_TYPE=Release"
        ], check=True)
        
        # Build the project
        subprocess.run([
            "cmake", 
            "--build", build_dir,
            "--config", "Release"
        ], check=True)
        
        print("C++ module built successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to build C++ module: {e}")
        return False
    except FileNotFoundError:
        print("CMake not found. Please install CMake to build the C++ module.")
        return False

# Try to build the C++ module
cpp_available = build_cpp_module()

if cpp_available:
    print("C++ MLPack engine is available!")
else:
    print("C++ MLPack engine is not available. Install dependencies:")
    print("  - CMake")
    print("  - MLPack")
    print("  - Armadillo")
    print("  - pybind11")

# Minimal setup for fallback
setup(
    name="least_squares_cpp",
    version="1.0.0",
    description="C++ implementation of least squares regression using MLPack",
    python_requires=">=3.7",
)