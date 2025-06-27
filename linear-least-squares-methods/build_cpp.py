#!/usr/bin/env python3
"""Standalone script to build the C++ module with better error reporting."""

import subprocess
import sys
import os
import shutil
import glob
import platform

def find_python_config():
    """Find python3-config executable."""
    candidates = ['python3-config', 'python-config', f'python{sys.version_info.major}.{sys.version_info.minor}-config']
    
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
    return None

def clean_previous_builds():
    """Clean up previous build artifacts."""
    print("🧹 Cleaning previous builds...")
    
    # Remove build directory
    build_dir = os.path.join('approaches', 'build')
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    
    # Remove any existing .so/.pyd files
    for pattern in ['*.so', '*.pyd', 'least_squares_cpp*.so', 'least_squares_cpp*.pyd']:
        for file in glob.glob(pattern):
            os.remove(file)
            print(f"  Removed: {file}")
        
        # Also check in approaches/ directory
        approaches_pattern = os.path.join('approaches', pattern)
        for file in glob.glob(approaches_pattern):
            os.remove(file)
            print(f"  Removed: {file}")

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n🔍 Checking dependencies...")
    
    missing = []
    
    # Check CMake
    if not shutil.which('cmake'):
        missing.append('cmake')
        print("  ❌ CMake not found")
    else:
        result = subprocess.run(['cmake', '--version'], capture_output=True, text=True)
        print(f"  ✅ CMake: {result.stdout.split()[2]}")
    
    # Check C++ compiler
    if not shutil.which('g++') and not shutil.which('clang++'):
        missing.append('g++ or clang++')
        print("  ❌ C++ compiler not found")
    else:
        compiler = 'g++' if shutil.which('g++') else 'clang++'
        result = subprocess.run([compiler, '--version'], capture_output=True, text=True)
        print(f"  ✅ C++ compiler: {compiler}")
    
    # Check Python development files
    python_config = find_python_config()
    if not python_config:
        missing.append('python3-dev')
        print("  ❌ Python development files not found")
    else:
        print(f"  ✅ Python config: {python_config}")
    
    # Check for Armadillo (harder to detect, so we'll try to compile a test)
    test_code = """
    #include <armadillo>
    int main() { arma::mat A(2,2); return 0; }
    """
    
    with open('test_armadillo.cpp', 'w') as f:
        f.write(test_code)
    
    try:
        compiler = 'g++' if shutil.which('g++') else 'clang++'
        result = subprocess.run([compiler, 'test_armadillo.cpp', '-larmadillo', '-o', 'test_armadillo'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ Armadillo library found")
            if os.path.exists('test_armadillo'):
                os.remove('test_armadillo')
        else:
            missing.append('libarmadillo-dev')
            print("  ❌ Armadillo library not found")
    except:
        missing.append('libarmadillo-dev')
        print("  ❌ Armadillo library not found")
    finally:
        if os.path.exists('test_armadillo.cpp'):
            os.remove('test_armadillo.cpp')
    
    # Check pybind11
    try:
        import pybind11
        print(f"  ✅ pybind11: {pybind11.__version__}")
    except ImportError:
        missing.append('python3-pybind11')
        print("  ❌ pybind11 not found")
    
    return missing

def build_cpp_module():
    """Build the C++ module with detailed output."""
    print("\n🔨 Building C++ module...")
    
    # Change to approaches directory
    os.chdir('approaches')
    
    # Create build directory
    os.makedirs('build', exist_ok=True)
    
    # Configure with CMake
    print("\n📋 Running CMake configuration...")
    cmake_args = [
        'cmake',
        '-S', '.',
        '-B', 'build',
        '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_VERBOSE_MAKEFILE=ON'
    ]
    
    result = subprocess.run(cmake_args, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ CMake configuration failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    print("✅ CMake configuration successful")
    
    # Build the project
    print("\n🏗️  Building project...")
    build_args = [
        'cmake',
        '--build', 'build',
        '--config', 'Release',
        '--verbose'
    ]
    
    result = subprocess.run(build_args, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Build failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    print("✅ Build successful")
    
    return True

def find_and_copy_module():
    """Find the compiled module and copy it to the right location."""
    print("\n📦 Looking for compiled module...")
    
    # Look for the module in build directory and subdirectories
    patterns = [
        'build/least_squares_cpp*.so',
        'build/*/least_squares_cpp*.so',
        'build/least_squares_cpp*.pyd',
        'build/*/least_squares_cpp*.pyd',
        'build/Release/least_squares_cpp*.pyd',
        'build/Debug/least_squares_cpp*.pyd'
    ]
    
    found_modules = []
    for pattern in patterns:
        found_modules.extend(glob.glob(pattern))
    
    if not found_modules:
        print("❌ No compiled module found!")
        return False
    
    # Use the first found module
    module_path = found_modules[0]
    print(f"  Found: {module_path}")
    
    # Copy to approaches directory
    module_name = os.path.basename(module_path)
    shutil.copy2(module_path, module_name)
    print(f"  Copied to: ./approaches/{module_name}")
    
    # Also copy to main directory (go back one level)
    os.chdir('..')
    shutil.copy2(os.path.join('approaches', module_name), module_name)
    print(f"  Copied to: ./{module_name}")
    
    return True

def test_import():
    """Test if the module can be imported."""
    print("\n🧪 Testing import...")
    
    try:
        # Try importing from current directory
        import least_squares_cpp
        print("✅ Module imported successfully from current directory!")
        
        # Test basic functionality
        model = least_squares_cpp.LinearRegression(2)
        print("✅ Created LinearRegression model successfully!")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        
        # Try from approaches directory
        try:
            sys.path.insert(0, 'approaches')
            import least_squares_cpp
            print("✅ Module imported successfully from approaches directory!")
            return True
        except ImportError as e:
            print(f"❌ Import from approaches also failed: {e}")
            return False

def main():
    """Main build process."""
    print("🚀 C++ Module Build Script")
    print("=" * 50)
    
    # Store original directory
    original_dir = os.getcwd()
    
    try:
        # Check dependencies
        missing = check_dependencies()
        if missing:
            print("\n❌ Missing dependencies:")
            for dep in missing:
                print(f"  - {dep}")
            print("\nInstall missing dependencies and try again.")
            
            if sys.platform.startswith('linux'):
                print("\nOn Ubuntu/Debian, try:")
                print("  sudo apt-get install cmake libarmadillo-dev python3-pybind11 python3-dev g++")
            elif sys.platform == 'darwin':
                print("\nOn macOS with Homebrew, try:")
                print("  brew install cmake armadillo pybind11")
            
            return 1
        
        # Clean previous builds
        clean_previous_builds()
        
        # Build the module
        if not build_cpp_module():
            print("\n❌ Build failed! Check the error messages above.")
            return 1
        
        # Find and copy the module
        if not find_and_copy_module():
            print("\n❌ Could not find compiled module!")
            return 1
        
        # Test import
        if not test_import():
            print("\n❌ Module built but cannot be imported!")
            print("\nTry running Python from the project directory:")
            print("  cd", os.getcwd())
            print("  python3 -c 'import least_squares_cpp'")
            return 1
        
        print("\n✅ Success! C++ module is ready to use.")
        print("\nYou can now run: python main.py")
        return 0
        
    finally:
        # Return to original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    sys.exit(main())