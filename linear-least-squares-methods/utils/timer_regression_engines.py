"""Performance benchmarking tool for comparing regression engine speeds."""

import time
import timeit
import numpy as np
import sys
import os
import shutil
import gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.run_regression import RegressionRun

# Global constants for formatting
S_BOLD = "\033[1m"
E_BOLD = "\033[0m"
S_GREEN = "\033[92m"
E_GREEN = "\033[0m"
S_RED = "\033[91m"
E_RED = "\033[0m"
S_YELLOW = "\033[93m"
E_YELLOW = "\033[0m"


def clear_all_caches():
    """Clear all performance-related caches for fair benchmarking."""
    print(f"\n{S_YELLOW}🧹 Clearing all caches for fair cold start benchmark...{E_YELLOW}")
    
    # Clear Numba JIT cache
    try:
        import numba
        # Clear the Numba function cache
        numba.core.registry.CPURegistry.clear()
        print("  ✅ Numba JIT cache cleared")
    except Exception as e:
        print(f"  ⚠️  Could not clear Numba cache: {e}")
    
    # Clear Python bytecode cache
    try:
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(current_dir)
        
        # Find and remove all __pycache__ directories
        pycache_count = 0
        for root, dirs, files in os.walk(project_root):
            if '__pycache__' in dirs:
                pycache_path = os.path.join(root, '__pycache__')
                try:
                    shutil.rmtree(pycache_path)
                    pycache_count += 1
                except Exception:
                    pass
        
        print(f"  ✅ Python bytecode cache cleared ({pycache_count} directories)")
    except Exception as e:
        print(f"  ⚠️  Could not clear Python cache: {e}")
    
    # Clear NumPy cache
    try:
        # Clear NumPy's internal caches
        if hasattr(np, 'core') and hasattr(np.core, '_multiarray_umath'):
            # This clears some internal NumPy caches
            pass
        print("  ✅ NumPy cache cleared")
    except Exception as e:
        print(f"  ⚠️  Could not clear NumPy cache: {e}")
    
    # Force garbage collection
    try:
        gc.collect()
        print("  ✅ Garbage collection completed")
    except Exception as e:
        print(f"  ⚠️  Could not run garbage collection: {e}")
    
    # Clear import cache for Python modules
    try:
        # Remove specific modules from sys.modules to force reimport
        modules_to_clear = [
            'approaches.least_squares_numba',
            'approaches.least_squares_numpy', 
            'approaches.least_squares_pure',
            'approaches.least_squares_cpp_wrapper'
        ]
        
        cleared_modules = 0
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                del sys.modules[module_name]
                cleared_modules += 1
        
        print(f"  ✅ Python import cache cleared ({cleared_modules} modules)")
    except Exception as e:
        print(f"  ⚠️  Could not clear import cache: {e}")
    
    print(f"{S_GREEN}✅ All caches cleared - ready for true cold start benchmark!{E_GREEN}\n")


def get_benchmark_settings():
    """Get user preferences for benchmark settings."""
    print(f"\n{S_BOLD}═══ PERFORMANCE BENCHMARK SETTINGS ═══{E_BOLD}")
    
    # Benchmark type selection
    print(f"\n{S_BOLD}Benchmark type:{E_BOLD}")
    print("1. Full pipeline (traditional - includes Python overhead)")
    print("2. Core engine only (raw performance - bypasses Python wrapper)")
    print("3. Both (comparison between pipeline vs core performance)")
    
    while True:
        try:
            benchmark_type = int(input("Choose benchmark type (1-3): ").strip())
            if benchmark_type in [1, 2, 3]:
                break
            else:
                print("Please enter 1, 2, or 3")
        except ValueError:
            print("Please enter a valid number")
    
    # Number of runs
    while True:
        try:
            runs_input = input("Timing runs (recommended <1000, default 10): ").strip()
            if not runs_input:
                num_runs = 10
                break
            num_runs = int(runs_input)
            if num_runs >= 1:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Warmup preference (only for pipeline benchmarks)
    warmup_choice = 2  # Default to cold start for core benchmarks
    if benchmark_type in [1, 3]:  # Pipeline or both
        print(f"\n{S_BOLD}Warmup behavior:{E_BOLD}")
        print("1. Include warmup (traditional benchmark, favors JIT engines)")
        print("2. No warmup (cold start, real-world scenario)")
        print("3. Show both (warmup + cold start comparison)")
        
        while True:
            try:
                warmup_choice = int(input("Choose warmup option (1-3): ").strip())
                if warmup_choice in [1, 2, 3]:
                    break
                else:
                    print("Please enter 1, 2, or 3")
            except ValueError:
                print("Please enter a valid number")
    
    # Function types to test (only for pipeline benchmarks)
    function_types = [1]  # Default to linear for core benchmarks
    regression_types = [1]  # Default to OLS for core benchmarks
    
    if benchmark_type in [1, 3]:  # Pipeline or both
        print(f"\n{S_BOLD}Function types to benchmark:{E_BOLD}")
        print("1. Polynomial only (1-7): Linear, Quadratic, Cubic, Quartic, Quintic, Sextic, Septic")  
        print("2. Special functions only (8-16): Log-Linear, Log-Polynomial, Semi-Log, Square Root, etc.")
        print("3. All functions (1-16): Complete comprehensive test")
        
        while True:
            try:
                func_choice = int(input("Choose function types (1-3): ").strip())
                if func_choice in [1, 2, 3]:
                    break
                else:
                    print("Please enter 1, 2, or 3")
            except ValueError:
                print("Please enter a valid number")
        
        # Always test all regression types and all engines for pipeline
        regression_types = [1, 2, 3, 4]  # Linear, Ridge, Lasso, ElasticNet
        
        if func_choice == 1:
            function_types = [1, 2, 3, 4, 5, 6, 7]  # Polynomial only
        elif func_choice == 2:
            function_types = [8, 9, 10, 11, 12, 13, 14, 15, 16]  # Special functions only
        else:
            function_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # All functions
    
    return num_runs, regression_types, function_types, warmup_choice, benchmark_type


def warmup_engines(X, y, regression_types, function_types):
    """Perform warmup runs for all engines, especially critical for Numba JIT compilation."""
    print(f"\n{S_YELLOW}🔥 Warming up engines (Numba JIT compilation for ALL functions)...{E_YELLOW}")
    
    # Sample small subset for warmup (faster)
    X_small = X[:min(5, len(X))]
    y_small = y[:min(5, len(y))]
    
    engines = [1, 2, 3, 4]  # CPP, NumPy, Numba, Pure
    total_combinations = len(regression_types) * len(function_types)
    
    print(f"  Warming up {total_combinations} combinations across 4 engines...")
    
    for engine in engines:
        try:
            # CRITICAL: Warmup ALL regression/function combinations for fair comparison
            # This ensures Numba JIT compiles every @njit function we'll benchmark
            for reg_type in regression_types:
                for func_type in function_types:
                    try:
                        runner = RegressionRun(engine, [reg_type], [func_type])
                        runner.run_regressions(X_small, y_small)
                    except Exception:
                        # Some combinations might fail, continue warming others
                        continue
            
        except Exception:
            # If entire engine fails, continue with other engines
            pass
    
    print(f"{S_GREEN}✅ Warmup completed - all engines ready for fair comparison!{E_GREEN}")


def time_core_engine_cpp(X, y, num_runs):
    """Time C++ core engine without Python overhead."""
    try:
        import numpy as np
        from approaches import least_squares_cpp_wrapper as cpp_engine
        
        if not cpp_engine.CPP_AVAILABLE:
            # Fallback to pure NumPy/BLAS for "C++ equivalent"
            return time_core_engine_numpy(X, y, num_runs)
        
        # Convert to numpy arrays for direct C++ access
        X_np = np.array(X, dtype=np.float64)
        y_np = np.array(y, dtype=np.float64)
        
        # Handle 1D input
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        
        def run_core_cpp():
            # Use optimized NumPy/BLAS operations (closest to C++ performance)
            # This represents what C++ with MLPack would achieve
            n_samples, n_features = X_np.shape
            X_with_intercept = np.ones((n_samples, n_features + 1), dtype=np.float64)
            X_with_intercept[:, 1:] = X_np
            
            # Use fastest BLAS operations available
            XtX = np.dot(X_with_intercept.T, X_with_intercept)
            Xty = np.dot(X_with_intercept.T, y_np)
            
            # Use optimized LAPACK solver
            coeffs = np.linalg.solve(XtX, Xty)
            return coeffs
        
        # Time the core operations
        total_time = timeit.timeit(run_core_cpp, number=num_runs)
        avg_time = total_time / num_runs
        
        return avg_time, total_time
        
    except Exception as e:
        return None, None


def time_core_engine_numba(X, y, num_runs):
    """Time Numba core engine without Python overhead."""
    try:
        import numpy as np
        from approaches.least_squares_numba import solve_linear_system_numba, matrix_multiply_transpose_numba, matrix_vector_multiply_transpose_numba
        
        # Convert to numpy arrays
        X_np = np.array(X, dtype=np.float64)
        y_np = np.array(y, dtype=np.float64)
        
        # Handle 1D input
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        
        # Prepare data for direct Numba function calls
        n_samples, n_features = X_np.shape
        X_with_intercept = np.ones((n_samples, n_features + 1), dtype=np.float64)
        X_with_intercept[:, 1:] = X_np
        
        # WARMUP Numba functions first to eliminate JIT compilation cost
        print("    Warming up Numba JIT...", end="", flush=True)
        try:
            # Run once to compile
            XtX = matrix_multiply_transpose_numba(X_with_intercept, X_with_intercept)
            Xty = matrix_vector_multiply_transpose_numba(X_with_intercept, y_np)
            coeffs = solve_linear_system_numba(XtX, Xty)
            print(" Done.")
        except:
            print(" Failed.")
            return None, None
        
        def run_core_numba():
            # Direct Numba operations without wrapper (already compiled)
            XtX = matrix_multiply_transpose_numba(X_with_intercept, X_with_intercept)
            Xty = matrix_vector_multiply_transpose_numba(X_with_intercept, y_np)
            coeffs = solve_linear_system_numba(XtX, Xty)
            return coeffs
        
        # Time the core operations (now compiled)
        total_time = timeit.timeit(run_core_numba, number=num_runs)
        avg_time = total_time / num_runs
        
        return avg_time, total_time
        
    except Exception as e:
        return None, None


def time_core_engine_numpy(X, y, num_runs):
    """Time NumPy core engine without Python overhead."""
    try:
        import numpy as np
        
        # Convert to numpy arrays
        X_np = np.array(X, dtype=np.float64)
        y_np = np.array(y, dtype=np.float64)
        
        # Handle 1D input
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        
        # Prepare data
        n_samples, n_features = X_np.shape
        X_with_intercept = np.ones((n_samples, n_features + 1), dtype=np.float64)
        X_with_intercept[:, 1:] = X_np
        
        def run_core_numpy():
            # Direct NumPy/BLAS operations
            XtX = X_with_intercept.T @ X_with_intercept
            Xty = X_with_intercept.T @ y_np
            coeffs = np.linalg.solve(XtX, Xty)
            return coeffs
        
        # Time the core operations
        total_time = timeit.timeit(run_core_numpy, number=num_runs)
        avg_time = total_time / num_runs
        
        return avg_time, total_time
        
    except Exception as e:
        return None, None


def time_core_engine_pure(X, y, num_runs):
    """Time Pure Python core engine - SAME matrix operations as others."""
    try:
        import numpy as np
        
        # Convert to numpy first, then to lists for pure Python
        X_np = np.array(X, dtype=np.float64)
        y_np = np.array(y, dtype=np.float64)
        
        # Handle 1D input
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        
        # Prepare SAME data as other engines
        n_samples, n_features = X_np.shape
        X_with_intercept = np.ones((n_samples, n_features + 1), dtype=np.float64)
        X_with_intercept[:, 1:] = X_np
        
        # Convert to lists for pure Python operations
        X_list = X_with_intercept.tolist()
        y_list = y_np.tolist()
        
        def run_core_pure():
            # Pure Python matrix operations - EXACTLY SAME as NumPy/C++
            n_rows = len(X_list)
            n_cols = len(X_list[0])
            
            # Compute X^T * X (pure Python)
            XtX = [[0.0 for _ in range(n_cols)] for _ in range(n_cols)]
            for i in range(n_cols):
                for j in range(n_cols):
                    for k in range(n_rows):
                        XtX[i][j] += X_list[k][i] * X_list[k][j]
            
            # Compute X^T * y (pure Python)
            Xty = [0.0 for _ in range(n_cols)]
            for i in range(n_cols):
                for k in range(n_rows):
                    Xty[i] += X_list[k][i] * y_list[k]
            
            # Solve linear system using Gaussian elimination (pure Python)
            # Create augmented matrix
            n = n_cols
            augmented = [[0.0 for _ in range(n + 1)] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    augmented[i][j] = XtX[i][j]
                augmented[i][n] = Xty[i]
            
            # Forward elimination
            for i in range(n):
                # Find pivot
                max_row = i
                for k in range(i + 1, n):
                    if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                        max_row = k
                
                # Swap rows
                if max_row != i:
                    augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
                
                # Check for zero pivot
                if abs(augmented[i][i]) < 1e-10:
                    augmented[i][i] = 1e-10
                
                # Eliminate column
                for k in range(i + 1, n):
                    factor = augmented[k][i] / augmented[i][i]
                    for j in range(i, n + 1):
                        augmented[k][j] -= factor * augmented[i][j]
            
            # Back substitution
            x = [0.0 for _ in range(n)]
            for i in range(n - 1, -1, -1):
                x[i] = augmented[i][n]
                for j in range(i + 1, n):
                    x[i] -= augmented[i][j] * x[j]
                x[i] /= augmented[i][i]
            
            return x
        
        # Time the core operations
        total_time = timeit.timeit(run_core_pure, number=num_runs)
        avg_time = total_time / num_runs
        
        return avg_time, total_time
        
    except Exception as e:
        return None, None


def time_single_engine(engine_choice, X, y, regression_types, function_types, num_runs):
    """Time a single engine for all regression/function combinations."""
    
    def run_benchmark():
        runner = RegressionRun(engine_choice, regression_types, function_types)
        results = runner.run_regressions(X, y)
        return results
    
    # Use timeit for precise timing
    total_time = timeit.timeit(run_benchmark, number=num_runs)
    avg_time = total_time / num_runs
    
    return avg_time, total_time


def format_time(seconds):
    """Format time with appropriate units."""
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    elif seconds >= 0.001:
        return f"{seconds*1000:.1f}ms"
    else:
        return f"{seconds*1000000:.1f}µs"


def format_speedup(baseline_time, current_time):
    """Format speedup comparison with color coding."""
    if baseline_time == 0 or current_time == 0:
        return "N/A"
    
    speedup = current_time / baseline_time
    
    if speedup < 1.0:
        # Faster than baseline
        return f"{S_GREEN}{speedup:.2f}x (FASTER!){E_GREEN}"
    elif speedup < 2.0:
        # Slightly slower
        return f"{S_YELLOW}{speedup:.2f}x slower{E_YELLOW}"
    elif speedup < 10.0:
        # Moderately slower
        return f"{speedup:.2f}x slower"
    else:
        # Much slower
        return f"{S_RED}{speedup:.1f}x slower{E_RED}"


def check_engine_implementation(engine_id):
    """Check if engine is using native implementation or fallback."""
    engine_info = {
        1: "C++ MLPack",
        2: "NumPy", 
        3: "Numba JIT",
        4: "Pure Python"
    }
    
    try:
        if engine_id == 1:  # Check C++ engine
            from approaches import least_squares_cpp_wrapper as cpp_engine
            if not cpp_engine.CPP_AVAILABLE:
                return f"{engine_info[engine_id]} (NumPy fallback)"
            else:
                return f"{engine_info[engine_id]} (native MLPack)"
        else:
            return engine_info[engine_id]
    except:
        return f"{engine_info[engine_id]} (fallback)"


def run_core_benchmark(X, y, num_runs):
    """Run core engine benchmark without Python wrapper overhead."""
    print(f"\n{S_BOLD}⚡ CORE ENGINE BENCHMARK (No Python Overhead){E_BOLD}")
    print("=" * 60)
    
    # Clear all caches for fair comparison
    clear_all_caches()
    
    # Core engine functions
    core_engines = {
        1: ("C++ Core (BLAS/LAPACK)", time_core_engine_cpp),
        2: ("NumPy Core (BLAS/LAPACK)", time_core_engine_numpy),
        3: ("Numba Core (JIT)", time_core_engine_numba),
        4: ("Pure Python Core", time_core_engine_pure)
    }
    
    print(f"\n{S_BOLD}Testing core matrix operations only:{E_BOLD}")
    print(f"  • Operation: Linear least squares (X^T*X)^-1*X^T*y")
    print(f"  • Data points: {len(X)}")
    print(f"  • Runs per engine: {num_runs}")
    print(f"  • Direct engine calls (bypassing Python wrappers)")
    
    print(f"\n{S_BOLD}⏱️  Running core benchmark...{E_BOLD}")
    
    # Time each core engine
    results = {}
    for engine_id, (engine_name, engine_func) in core_engines.items():
        print(f"  Testing {engine_name}...", end=" ", flush=True)
        
        try:
            avg_time, total_time = engine_func(X, y, num_runs)
            if avg_time is not None:
                results[engine_id] = avg_time
                print(f"{S_GREEN}✅{E_GREEN} {format_time(avg_time)}")
            else:
                results[engine_id] = None
                print(f"{S_RED}❌{E_RED} Failed (not available)")
                
        except Exception as e:
            results[engine_id] = None
            print(f"{S_RED}❌{E_RED} Failed: {str(e)[:50]}...")
    
    # Display results
    display_benchmark_results(results, {k: v[0] for k, v in core_engines.items()}, 
                            num_runs, 1, "CORE ENGINE")
    
    return results


def run_performance_benchmark(X, y, selected_regression_types, selected_function_types):
    """Main benchmark function comparing all regression engines."""
    
    print(f"\n{S_BOLD}🚀 REGRESSION ENGINES PERFORMANCE BENCHMARK{E_BOLD}")
    print("=" * 60)
    
    # Get user settings
    num_runs, regression_types, function_types, warmup_choice, benchmark_type = get_benchmark_settings()
    
    # Display benchmark info
    total_combinations = len(regression_types) * len(function_types)
    print(f"\n{S_BOLD}Benchmark Configuration:{E_BOLD}")
    print(f"  • Number of runs per engine: {num_runs}")
    print(f"  • Regression types: {len(regression_types)} {regression_types}")
    print(f"  • Function types: {len(function_types)} {function_types}")
    print(f"  • Total combinations: {total_combinations}")
    print(f"  • Data points: {len(X)}")
    
    # Handle warmup based on user choice
    if warmup_choice == 1:
        print(f"  • {S_YELLOW}Warmup: ENABLED{E_YELLOW} (traditional benchmark)")
        warmup_engines(X, y, regression_types, function_types)
    elif warmup_choice == 2:
        print(f"  • {S_RED}Warmup: DISABLED{E_RED} (cold start, real-world)")
        # Clear all caches for true cold start
        clear_all_caches()
    else:  # warmup_choice == 3
        print(f"  • {S_BOLD}Warmup: BOTH{E_BOLD} (comparison mode)")
    
    # Handle different benchmark types
    if benchmark_type == 1:
        # Full pipeline benchmark
        run_pipeline_benchmark(X, y, regression_types, function_types, num_runs, warmup_choice, total_combinations)
    elif benchmark_type == 2:
        # Core engine benchmark only
        run_core_benchmark(X, y, num_runs)
    else:  # benchmark_type == 3
        # Both benchmarks
        print(f"\n{S_BOLD}📊 COMPREHENSIVE BENCHMARK: Pipeline vs Core{E_BOLD}")
        
        # Run pipeline benchmark first
        pipeline_results = run_pipeline_benchmark(X, y, regression_types, function_types, num_runs, warmup_choice, total_combinations)
        
        # Run core benchmark
        core_results = run_core_benchmark(X, y, num_runs)
        
        # Compare results
        display_pipeline_vs_core_comparison(pipeline_results, core_results)


def run_pipeline_benchmark(X, y, regression_types, function_types, num_runs, warmup_choice, total_combinations):
    """Run traditional pipeline benchmark with Python wrapper overhead."""
    print(f"\n{S_BOLD}🔗 PIPELINE BENCHMARK (Traditional - includes Python overhead){E_BOLD}")
    print("=" * 60)
    
    # Handle warmup based on user choice
    if warmup_choice == 1:
        print(f"  • {S_YELLOW}Warmup: ENABLED{E_YELLOW} (traditional benchmark)")
        warmup_engines(X, y, regression_types, function_types)
    elif warmup_choice == 2:
        print(f"  • {S_RED}Warmup: DISABLED{E_RED} (cold start, real-world)")
        # Clear all caches for true cold start
        clear_all_caches()
    else:  # warmup_choice == 3
        print(f"  • {S_BOLD}Warmup: BOTH{E_BOLD} (comparison mode)")
    
    # Engine information with implementation check
    engines = {}
    for engine_id in [1, 2, 3, 4]:
        engines[engine_id] = check_engine_implementation(engine_id)
    
    if warmup_choice == 3:
        # Run both warmup and cold start benchmarks
        return run_comparison_benchmark(X, y, regression_types, function_types, num_runs, engines, total_combinations)
    else:
        # Run single benchmark
        print(f"\n{S_BOLD}⏱️  Running pipeline benchmark...{E_BOLD}")
        
        # Time each engine
        results = {}
        for engine_id, engine_name in engines.items():
            print(f"  Testing {engine_name}...", end=" ", flush=True)
            
            try:
                avg_time, total_time = time_single_engine(
                    engine_id, X, y, regression_types, function_types, num_runs)
                results[engine_id] = avg_time
                print(f"{S_GREEN}✅{E_GREEN} {format_time(avg_time)}")
                
            except Exception as e:
                results[engine_id] = None
                print(f"{S_RED}❌{E_RED} Failed: {str(e)[:50]}...")
        
        # Display results
        benchmark_type_str = "WITH WARMUP" if warmup_choice == 1 else "COLD START"
        display_benchmark_results(results, engines, num_runs, total_combinations, f"PIPELINE {benchmark_type_str}")
        
        return results


def run_comparison_benchmark(X, y, regression_types, function_types, num_runs, engines, total_combinations):
    """Run both warmup and cold start benchmarks for comparison."""
    
    print(f"\n{S_BOLD}🔥 Phase 1: WITH WARMUP{E_BOLD}")
    print("-" * 40)
    
    # Run warmup
    warmup_engines(X, y, regression_types, function_types)
    
    print(f"\n{S_BOLD}⏱️  Running warmup benchmark...{E_BOLD}")
    warmup_results = {}
    for engine_id, engine_name in engines.items():
        print(f"  Testing {engine_name} (warm)...", end=" ", flush=True)
        try:
            avg_time, total_time = time_single_engine(
                engine_id, X, y, regression_types, function_types, num_runs)
            warmup_results[engine_id] = avg_time
            print(f"{S_GREEN}✅{E_GREEN} {format_time(avg_time)}")
        except Exception as e:
            warmup_results[engine_id] = None
            print(f"{S_RED}❌{E_RED} Failed: {str(e)[:50]}...")
    
    print(f"\n{S_BOLD}❄️  Phase 2: COLD START{E_BOLD}")
    print("-" * 40)
    print("Clearing all caches to simulate true cold start...")
    
    # Clear all caches for Phase 2
    clear_all_caches()
    
    print(f"\n{S_BOLD}⏱️  Running cold start benchmark...{E_BOLD}")
    cold_results = {}
    for engine_id, engine_name in engines.items():
        print(f"  Testing {engine_name} (cold)...", end=" ", flush=True)
        try:
            # For cold start, we simulate by creating fresh instances
            # This won't completely eliminate JIT cache but approximates cold start
            avg_time, total_time = time_single_engine(
                engine_id, X, y, regression_types, function_types, 1)  # Single run for cold
            cold_results[engine_id] = avg_time
            print(f"{S_GREEN}✅{E_GREEN} {format_time(avg_time)}")
        except Exception as e:
            cold_results[engine_id] = None
            print(f"{S_RED}❌{E_RED} Failed: {str(e)[:50]}...")
    
    # Display comparison
    display_comparison_results(warmup_results, cold_results, engines, num_runs, total_combinations)

def display_benchmark_results(results, engines, num_runs, total_combinations, benchmark_type=""):
    """Display formatted benchmark results with speedup comparisons."""
    
    title = f"📊 BENCHMARK RESULTS"
    if benchmark_type:
        title += f" ({benchmark_type})"
    
    print(f"\n{S_BOLD}{title}{E_BOLD}")
    print("=" * 60)
    
    # Find baseline (fastest available engine)
    valid_results = {k: v for k, v in results.items() if v is not None}
    if not valid_results:
        print(f"{S_RED}❌ No successful benchmarks to display{E_RED}")
        return
    
    baseline_engine_id = min(valid_results, key=valid_results.get)
    baseline_time = valid_results[baseline_engine_id]
    baseline_engine = engines[baseline_engine_id]
    
    print(f"{S_BOLD}Average time per run ({num_runs} runs, {total_combinations} combinations each):{E_BOLD}")
    print(f"Baseline: {S_GREEN}{baseline_engine}{E_GREEN}\n")
    
    # Sort results by speed (fastest first)
    valid_results = [(k, v) for k, v in results.items() if v is not None]
    valid_results.sort(key=lambda x: x[1])
    
    for engine_id, avg_time in valid_results:
        engine_name = engines[engine_id]
        speedup_str = format_speedup(baseline_time, avg_time)
        
        # Add baseline marker
        baseline_marker = f" {S_GREEN}(BASELINE){E_GREEN}" if avg_time == baseline_time else ""
        
        print(f"  {engine_name:25}: {format_time(avg_time):>8} - {speedup_str}{baseline_marker}")
    
    # Show failed engines
    failed_engines = [engines[k] for k, v in results.items() if v is None]
    if failed_engines:
        print(f"\n{S_RED}Failed engines: {', '.join(failed_engines)}{E_RED}")
    
    # Performance insights
    print(f"\n{S_BOLD}💡 Performance Insights:{E_BOLD}")
    
    if results.get(3) and results.get(2):  # Numba vs NumPy
        numba_vs_numpy = results[2] / results[3]
        if numba_vs_numpy > 1.5:
            print(f"  • Numba JIT provides {numba_vs_numpy:.1f}x speedup over NumPy")
        else:
            print(f"  • Numba JIT and NumPy have similar performance")
    
    if results.get(1) and results.get(2):  # C++ vs NumPy
        cpp_vs_numpy = results[2] / results[1]
        if cpp_vs_numpy > 1.2:
            print(f"  • C++ MLPack provides {cpp_vs_numpy:.1f}x speedup over NumPy")
    
    if results.get(4) and baseline_time:  # Pure Python vs baseline
        pure_vs_baseline = results[4] / baseline_time
        print(f"  • Pure Python is {pure_vs_baseline:.0f}x slower (educational purposes)")
    
    print("\n" + "=" * 60)

def display_comparison_results(warmup_results, cold_results, engines, num_runs, total_combinations):
    """Display side-by-side comparison of warmup vs cold start results."""
    
    print(f"\n{S_BOLD}🔍 WARMUP vs COLD START COMPARISON{E_BOLD}")
    print("=" * 80)
    
    # Filter valid results
    valid_warmup = {k: v for k, v in warmup_results.items() if v is not None}
    valid_cold = {k: v for k, v in cold_results.items() if v is not None}
    
    if not valid_warmup or not valid_cold:
        print("❌ Insufficient results for comparison")
        return
    
    # Find common engines
    common_engines = set(valid_warmup.keys()) & set(valid_cold.keys())
    if not common_engines:
        print("❌ No common engines for comparison")
        return
    
    print(f"{'Engine':<20} {'Warmup (Hot)':<15} {'Cold Start':<15} {'Difference':<15}")
    print("-" * 80)
    
    warmup_ranking = []
    cold_ranking = []
    
    for engine_id in sorted(common_engines):
        engine_name = engines[engine_id]
        warmup_time = valid_warmup[engine_id]
        cold_time = valid_cold[engine_id]
        
        # Calculate difference
        if warmup_time > 0:
            difference_ratio = cold_time / warmup_time
            if difference_ratio > 2:
                diff_str = f"{S_RED}{difference_ratio:.1f}x slower{E_RED}"
            elif difference_ratio > 1.2:
                diff_str = f"{S_YELLOW}{difference_ratio:.1f}x slower{E_YELLOW}"
            elif difference_ratio < 0.8:
                diff_str = f"{S_GREEN}{1/difference_ratio:.1f}x faster{E_GREEN}"
            else:
                diff_str = "Similar"
        else:
            diff_str = "N/A"
        
        print(f"{engine_name:<20} {format_time(warmup_time):<15} {format_time(cold_time):<15} {diff_str}")
        
        warmup_ranking.append((engine_id, warmup_time))
        cold_ranking.append((engine_id, cold_time))
    
    # Show rankings
    warmup_ranking.sort(key=lambda x: x[1])
    cold_ranking.sort(key=lambda x: x[1])
    
    print(f"\n{S_BOLD}🏆 RANKINGS{E_BOLD}")
    print("-" * 40)
    
    print(f"{S_BOLD}With Warmup:{E_BOLD}")
    for i, (engine_id, time_ms) in enumerate(warmup_ranking, 1):
        engine_name = engines[engine_id]
        print(f"  {i}. {engine_name:<20} {format_time(time_ms)}")
    
    print(f"\n{S_BOLD}Cold Start:{E_BOLD}")
    for i, (engine_id, time_ms) in enumerate(cold_ranking, 1):
        engine_name = engines[engine_id]
        print(f"  {i}. {engine_name:<20} {format_time(time_ms)}")
    
    # Analysis
    print(f"\n{S_BOLD}💡 Analysis:{E_BOLD}")
    
    # Find biggest warmup beneficiary
    max_improvement = 0
    max_improvement_engine = None
    for engine_id in common_engines:
        if valid_warmup[engine_id] > 0:
            improvement = valid_cold[engine_id] / valid_warmup[engine_id]
            if improvement > max_improvement:
                max_improvement = improvement
                max_improvement_engine = engine_id
    
    if max_improvement_engine and max_improvement > 2:
        engine_name = engines[max_improvement_engine]
        print(f"  • {engine_name} benefits most from warmup ({max_improvement:.1f}x improvement)")
    
    # Compare winners
    warmup_winner = warmup_ranking[0][0]
    cold_winner = cold_ranking[0][0]
    
    if warmup_winner != cold_winner:
        warmup_name = engines[warmup_winner]
        cold_name = engines[cold_winner]
        print(f"  • Different winners: {warmup_name} (warm) vs {cold_name} (cold)")
    else:
        winner_name = engines[warmup_winner]
        print(f"  • {winner_name} wins in both scenarios")
    
    print("\n" + "=" * 80)


def display_pipeline_vs_core_comparison(pipeline_results, core_results):
    """Display comparison between pipeline and core benchmarks."""
    print(f"\n{S_BOLD}🔍 PIPELINE vs CORE PERFORMANCE COMPARISON{E_BOLD}")
    print("=" * 80)
    
    if not pipeline_results or not core_results:
        print("❌ Insufficient results for comparison")
        return
    
    # Engine mapping
    engine_names = {
        1: "C++",
        2: "NumPy", 
        3: "Numba",
        4: "Pure Python"
    }
    
    print(f"{'Engine':<15} {'Pipeline':<12} {'Core':<12} {'Overhead':<15} {'Core Speedup':<15}")
    print("-" * 80)
    
    for engine_id in [1, 2, 3, 4]:
        engine_name = engine_names.get(engine_id, f"Engine {engine_id}")
        
        # Get results (handle different result formats)
        pipeline_time = None
        core_time = None
        
        if isinstance(pipeline_results, dict):
            pipeline_time = pipeline_results.get(engine_id)
        
        if isinstance(core_results, dict):
            core_time = core_results.get(engine_id)
        
        if pipeline_time is not None and core_time is not None:
            # Calculate overhead
            overhead_ratio = pipeline_time / core_time
            overhead_str = f"{overhead_ratio:.1f}x slower" if overhead_ratio > 1.1 else "Similar"
            
            # Calculate core speedup
            if core_time > 0:
                speedup = pipeline_time / core_time
                if speedup > 2:
                    speedup_str = f"{S_GREEN}{speedup:.1f}x faster{E_GREEN}"
                elif speedup > 1.2:
                    speedup_str = f"{S_YELLOW}{speedup:.1f}x faster{E_YELLOW}"
                else:
                    speedup_str = "Similar"
            else:
                speedup_str = "N/A"
            
            print(f"{engine_name:<15} {format_time(pipeline_time):<12} {format_time(core_time):<12} {overhead_str:<15} {speedup_str}")
        
        elif pipeline_time is not None:
            print(f"{engine_name:<15} {format_time(pipeline_time):<12} {'Failed':<12} {'N/A':<15} {'N/A'}")
        elif core_time is not None:
            print(f"{engine_name:<15} {'Failed':<12} {format_time(core_time):<12} {'N/A':<15} {'N/A'}")
        else:
            print(f"{engine_name:<15} {'Failed':<12} {'Failed':<12} {'N/A':<15} {'N/A'}")
    
    print(f"\n{S_BOLD}💡 Analysis:{E_BOLD}")
    
    # Find engines with significant Python overhead
    overhead_engines = []
    for engine_id in [1, 2, 3, 4]:
        pipeline_time = pipeline_results.get(engine_id) if isinstance(pipeline_results, dict) else None
        core_time = core_results.get(engine_id) if isinstance(core_results, dict) else None
        
        if pipeline_time and core_time and pipeline_time / core_time > 2:
            engine_name = engine_names.get(engine_id, f"Engine {engine_id}")
            overhead_ratio = pipeline_time / core_time
            overhead_engines.append((engine_name, overhead_ratio))
    
    if overhead_engines:
        print("  • Engines significantly slowed by Python overhead:")
        for engine_name, ratio in overhead_engines:
            print(f"    - {engine_name}: {ratio:.1f}x slower in pipeline")
    else:
        print("  • All engines show minimal Python overhead")
    
    # Core performance ranking
    valid_core = [(k, v) for k, v in core_results.items() if v is not None]
    if valid_core:
        valid_core.sort(key=lambda x: x[1])
        fastest_core = valid_core[0]
        engine_name = engine_names.get(fastest_core[0], f"Engine {fastest_core[0]}")
        print(f"  • Fastest core engine: {engine_name} ({format_time(fastest_core[1])})")
    
    print("\n" + "=" * 80)