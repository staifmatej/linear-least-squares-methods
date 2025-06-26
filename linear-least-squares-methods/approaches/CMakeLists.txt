cmake_minimum_required(VERSION 3.12)
project(least_squares_cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(pybind11 REQUIRED)
find_package(Armadillo REQUIRED)

# Find MLPack
find_package(PkgConfig REQUIRED)
pkg_check_modules(MLPACK REQUIRED mlpack)

# Add include directories
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${MLPACK_INCLUDE_DIRS})

# Create pybind11 module
pybind11_add_module(least_squares_cpp least_squares_cpp.cpp)

# Link libraries
target_link_libraries(least_squares_cpp PRIVATE ${ARMADILLO_LIBRARIES} ${MLPACK_LIBRARIES})
target_link_directories(least_squares_cpp PRIVATE ${MLPACK_LIBRARY_DIRS})
target_compile_options(least_squares_cpp PRIVATE ${MLPACK_CFLAGS_OTHER})

# Compiler-specific options
target_compile_definitions(least_squares_cpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Set properties
set_target_properties(least_squares_cpp PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
)