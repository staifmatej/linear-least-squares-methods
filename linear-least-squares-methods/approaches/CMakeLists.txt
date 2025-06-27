cmake_minimum_required(VERSION 3.12)
project(least_squares_cpp)

# Set C++ standard (MLPack requires C++17 minimum)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Try to find pybind11 - first as a package, then as a Python module
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(pybind11_DIR)
        find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})
    else()
        message(FATAL_ERROR "pybind11 not found. Install it with: pip install pybind11")
    endif()
endif()

# Find MLPack - try pkg-config first
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(MLPACK REQUIRED mlpack)
    set(MLPACK_FOUND TRUE)
    set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS})
    set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES})
else()
    find_package(mlpack REQUIRED)
    if(NOT mlpack_FOUND)
        message(FATAL_ERROR "MLPack not found. Install libmlpack-dev (Ubuntu) or mlpack (brew)")
    endif()
endif()

# Find Armadillo (required by MLPack)
find_package(Armadillo REQUIRED)
if(NOT ARMADILLO_FOUND)
    # Try pkg-config as fallback
    find_package(PkgConfig)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(ARMADILLO REQUIRED armadillo)
    else()
        message(FATAL_ERROR "Armadillo not found. Install libarmadillo-dev (Ubuntu) or armadillo (brew)")
    endif()
endif()

# Find Boost (required by MLPack)
find_package(Boost COMPONENTS program_options unit_test_framework serialization REQUIRED)

# Print debug information
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "pybind11 version: ${pybind11_VERSION}")
message(STATUS "MLPack version: ${mlpack_VERSION}")
message(STATUS "MLPack include dirs: ${MLPACK_INCLUDE_DIRS}")
message(STATUS "MLPack libraries: ${MLPACK_LIBRARIES}")
message(STATUS "Armadillo version: ${ARMADILLO_VERSION_STRING}")
message(STATUS "Armadillo include dirs: ${ARMADILLO_INCLUDE_DIRS}")
message(STATUS "Armadillo libraries: ${ARMADILLO_LIBRARIES}")
message(STATUS "Boost version: ${Boost_VERSION}")

# Create the module
pybind11_add_module(least_squares_cpp least_squares_cpp.cpp)

# Set properties
set_target_properties(least_squares_cpp PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

# Include directories
target_include_directories(least_squares_cpp PRIVATE
    ${MLPACK_INCLUDE_DIRS}
    ${ARMADILLO_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(least_squares_cpp PRIVATE
    ${MLPACK_LIBRARIES}
    ${ARMADILLO_LIBRARIES}
    ${Boost_LIBRARIES}
)

# Add explicit link directories
target_link_directories(least_squares_cpp PRIVATE
    ${MLPACK_LIBRARY_DIRS}
    /opt/homebrew/lib
)

# On some systems we need to explicitly link BLAS/LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
if(BLAS_FOUND AND LAPACK_FOUND)
    target_link_libraries(least_squares_cpp PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

# OpenMP support (MLPack can use it for parallelization)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(least_squares_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# Compiler-specific flags
if(MSVC)
    target_compile_options(least_squares_cpp PRIVATE /W4)
else()
    target_compile_options(least_squares_cpp PRIVATE -Wall -Wextra -Wpedantic)
    # Add optimization flags (removed -ffast-math due to infinity issues)
    target_compile_options(least_squares_cpp PRIVATE -O3 -march=native)

    # Enable AVX/SSE if available (for faster linear algebra)
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        target_compile_options(least_squares_cpp PRIVATE -mavx2)
    endif()
endif()

# Installation
install(TARGETS least_squares_cpp DESTINATION .)