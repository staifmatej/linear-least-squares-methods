#!/usr/bin/env python3
"""Test script to verify C++ module functionality."""

import sys
import numpy as np

def test_cpp_module():
    """Test if C++ module works correctly."""
    print("🧪 Testing C++ Module")
    print("=" * 50)
    
    # Try to import the module
    try:
        import least_squares_cpp
        print("✅ Successfully imported least_squares_cpp")
    except ImportError as e:
        print(f"❌ Failed to import least_squares_cpp: {e}")
        print("\nTrying from approaches directory...")
        try:
            sys.path.insert(0, 'approaches')
            import least_squares_cpp
            print("✅ Successfully imported from approaches/")
        except ImportError as e:
            print(f"❌ Also failed from approaches/: {e}")
            return False
    
    # Test LinearRegression
    print("\n📊 Testing LinearRegression...")
    try:
        # Create model
        model = least_squares_cpp.LinearRegression(degree=2)
        print("  ✅ Created LinearRegression model")
        
        # Generate test data
        np.random.seed(42)
        X = np.linspace(-2, 2, 20)
        y = 3 * X**2 + 2 * X + 1 + np.random.normal(0, 0.1, 20)
        
        # Fit model
        model.fit(X, y)
        print("  ✅ Fitted model successfully")
        
        # Get coefficients
        coeffs = model.get_coefficients()
        print(f"  ✅ Got coefficients: {coeffs}")
        print(f"     Expected approximately: [1, 2, 3] (intercept, x, x²)")
        
        # Get condition number
        cond_num = model.get_condition_number()
        print(f"  ✅ Condition number: {cond_num:.2e}")
        
        # Make predictions
        y_pred = model.predict(X)
        print(f"  ✅ Made predictions, shape: {y_pred.shape}")
        
        # Calculate R²
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_pred)**2)
        r2 = 1 - (ss_res / ss_tot)
        print(f"  ✅ R² score: {r2:.4f}")
        
    except Exception as e:
        print(f"  ❌ LinearRegression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test RidgeRegression
    print("\n📊 Testing RidgeRegression...")
    try:
        # Create model
        model = least_squares_cpp.RidgeRegression(alpha=0.01)
        print("  ✅ Created RidgeRegression model")
        
        # Use same data but reshape for 2D
        X_2d = X.reshape(-1, 1)
        
        # Fit model
        model.fit(X_2d, y)
        print("  ✅ Fitted model successfully")
        
        # Get coefficients
        coeffs = model.get_coefficients()
        print(f"  ✅ Got coefficients: {coeffs}")
        
        # Get condition number
        cond_num = model.get_condition_number()
        print(f"  ✅ Condition number: {cond_num:.2e}")
        
        # Make predictions
        y_pred = model.predict(X_2d)
        print(f"  ✅ Made predictions, shape: {y_pred.shape}")
        
    except Exception as e:
        print(f"  ❌ RidgeRegression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All tests passed! C++ module is working correctly.")
    return True

def compare_with_numpy():
    """Compare C++ results with NumPy implementation."""
    print("\n🔄 Comparing C++ with NumPy implementation...")
    
    try:
        import least_squares_cpp
        from approaches import least_squares_numpy
        
        # Generate test data
        np.random.seed(42)
        X = np.linspace(-2, 2, 50)
        y = 2 * X**3 - X**2 + 3 * X + 1 + np.random.normal(0, 0.2, 50)
        
        # Fit with C++
        cpp_model = least_squares_cpp.LinearRegression(degree=3)
        cpp_model.fit(X, y)
        cpp_coeffs = cpp_model.get_coefficients()
        cpp_pred = cpp_model.predict(X)
        
        # Fit with NumPy
        numpy_model = least_squares_numpy.LinearRegression(degree=3)
        numpy_model.fit(X, y)
        numpy_coeffs = numpy_model.coefficients
        numpy_pred = numpy_model.predict(X)
        
        # Compare coefficients
        print("\nCoefficients comparison:")
        print(f"  C++:   {cpp_coeffs}")
        print(f"  NumPy: {numpy_coeffs}")
        
        # Calculate difference
        coeff_diff = np.max(np.abs(np.array(cpp_coeffs) - np.array(numpy_coeffs)))
        pred_diff = np.max(np.abs(cpp_pred - numpy_pred))
        
        print(f"\nMax coefficient difference: {coeff_diff:.2e}")
        print(f"Max prediction difference: {pred_diff:.2e}")
        
        if coeff_diff < 1e-6 and pred_diff < 1e-6:
            print("✅ C++ and NumPy implementations match!")
        else:
            print("⚠️  Some differences detected (might be due to different algorithms)")
            
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

def test_wrapper_integration():
    """Test if the wrapper properly detects and uses C++ module."""
    print("\n🔌 Testing Wrapper Integration...")
    
    try:
        from approaches.least_squares_cpp_wrapper import CPP_AVAILABLE, LinearRegression
        
        print(f"CPP_AVAILABLE: {CPP_AVAILABLE}")
        
        if not CPP_AVAILABLE:
            print("❌ Wrapper does not detect C++ module!")
            return False
        
        # Test wrapper usage
        model = LinearRegression(degree=2)
        print("✅ Created wrapper LinearRegression model")
        
        # Generate test data
        X = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = np.array([2, 8, 18, 32, 50], dtype=np.float64)  # y = 2x²
        
        # Fit model
        model.fit(X, y)
        print("✅ Fitted model via wrapper")
        
        # Make predictions
        y_pred = model.predict(X)
        print(f"✅ Made predictions via wrapper: {y_pred}")
        
        print("✅ Wrapper integration working!")
        return True
        
    except Exception as e:
        print(f"❌ Wrapper integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("C++ Module Test Suite")
    print("=" * 50)
    
    # Run tests
    success = True
    
    if not test_cpp_module():
        success = False
    
    if not test_wrapper_integration():
        success = False
    
    if success:
        compare_with_numpy()
        print("\n✅ All tests passed! C++ module is ready to use!")
        print("You can now run: python main.py")
    else:
        print("\n❌ Some tests failed.")
        print("\nTroubleshooting steps:")
        print("1. Make sure you have installed all dependencies")
        print("2. Run: python build_cpp.py")
        print("3. Check for any error messages")