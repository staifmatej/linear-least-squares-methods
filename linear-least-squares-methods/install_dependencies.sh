#!/bin/bash

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "🐧 Detected Linux system"
    
    # Check if apt-get is available (Debian/Ubuntu)
    if command -v apt-get &> /dev/null; then
        echo "📦 Installing dependencies with apt-get..."
        sudo apt-get update
        sudo apt-get install -y \
            cmake \
            g++ \
            libarmadillo-dev \
            libblas-dev \
            liblapack-dev \
            python3-dev \
            python3-pybind11 \
            python3-pip
        
        # Install pybind11 via pip as backup
        pip3 install pybind11
        
    # Check if yum is available (RedHat/CentOS/Fedora)
    elif command -v yum &> /dev/null; then
        echo "📦 Installing dependencies with yum..."
        sudo yum install -y \
            cmake \
            gcc-c++ \
            armadillo-devel \
            blas-devel \
            lapack-devel \
            python3-devel \
            python3-pip
        
        # Install pybind11 via pip
        pip3 install pybind11
        
    # Check if dnf is available (newer Fedora)
    elif command -v dnf &> /dev/null; then
        echo "📦 Installing dependencies with dnf..."
        sudo dnf install -y \
            cmake \
            gcc-c++ \
            armadillo-devel \
            blas-devel \
            lapack-devel \
            python3-devel \
            python3-pip
        
        # Install pybind11 via pip
        pip3 install pybind11
        
    # Check if pacman is available (Arch Linux)
    elif command -v pacman &> /dev/null; then
        echo "📦 Installing dependencies with pacman..."
        sudo pacman -S --noconfirm \
            cmake \
            gcc \
            armadillo \
            blas \
            lapack \
            python \
            python-pip
        
        # Install pybind11 via pip
        pip install pybind11
        
    else
        echo "❌ Unknown Linux distribution. Please install manually:"
        echo "  - cmake"
        echo "  - g++ or clang++"
        echo "  - armadillo (linear algebra library)"
        echo "  - BLAS and LAPACK"
        echo "  - Python development headers"
        echo "  - pybind11"
        exit 1
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "🍎 Detected macOS"
    
    # Check if Homebrew is installed
    if command -v brew &> /dev/null; then
        echo "📦 Installing dependencies with Homebrew..."
        brew install cmake armadillo pybind11
        
        # Install pybind11 via pip as backup
        pip3 install pybind11
        
    else
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows
    echo "🪟 Detected Windows"
    echo "⚠️  Windows requires manual installation or WSL (Windows Subsystem for Linux)"
    echo ""
    echo "Option 1: Use WSL (recommended)"
    echo "  1. Install WSL: wsl --install"
    echo "  2. Open WSL terminal and run this script again"
    echo ""
    echo "Option 2: Use conda"
    echo "  conda install -c conda-forge cmake armadillo pybind11"
    echo ""
    echo "Option 3: Manual installation"
    echo "  - Install Visual Studio with C++ support"
    echo "  - Install CMake from https://cmake.org"
    echo "  - Install Armadillo from http://arma.sourceforge.net"
    echo "  - pip install pybind11"
    exit 1
    
else
    echo "❌ Unknown operating system: $OSTYPE"
    exit 1
fi

echo ""
echo "✅ Dependencies installed successfully!"
echo ""
echo "Next steps:"
echo "1. Run the build script: python3 build_cpp.py"
echo "2. If build succeeds, run: python3 main.py"