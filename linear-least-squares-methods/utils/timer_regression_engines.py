"""Performance benchmarking tool for comparing regression engine speeds."""

import time
import timeit
import numpy as np
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


def get_benchmark_settings():
    """Get user preferences for benchmark settings."""
    print(f"\n{S_BOLD}PPP PERFORMANCE BENCHMARK SETTINGS PPP{E_BOLD}")
    
    # Number of runs
    while True:
        try:
            runs_input = input("Enter number of timing runs (default 10, max 1000): ").strip()
            if not runs_input:
                num_runs = 10
                break
            num_runs = int(runs_input)
            if 1 <= num_runs <= 1000:
                break
            else:
                print("Please enter a number between 1 and 1000")
        except ValueError:
            print("Please enter a valid number")
    
    # Test scope
    print(f"\n{S_BOLD}Benchmark scope:{E_BOLD}")
    print("1. Test selected combinations only")  
    print("2. Test all regression types with selected functions")
    print("3. Test comprehensive (all regression � all polynomial functions 1-7)")
    
    while True:
        try:
            scope = int(input("Choose scope (1-3): ").strip())
            if scope in [1, 2, 3]:
                break
            else:
                print("Please enter 1, 2, or 3")
        except ValueError:
            print("Please enter a valid number")
    
    return num_runs, scope


def warmup_engines(X, y, regression_types, function_types):
    """Perform warmup runs for all engines, especially important for Numba JIT."""
    print(f"\n{S_YELLOW}=% Warming up engines (Numba JIT compilation)...{E_YELLOW}")
    
    # Sample small subset for warmup (faster)
    X_small = X[:min(10, len(X))]
    y_small = y[:min(10, len(y))]
    
    engines = [1, 2, 3, 4]  # CPP, NumPy, Numba, Pure
    
    for engine in engines:
        try:
            # Use first available regression and function type for warmup
            reg_type = regression_types[0] if regression_types else 1
            func_type = function_types[0] if function_types else 1
            
            runner = RegressionRun(engine, [reg_type], [func_type])
            runner.run_regressions(X_small, y_small)
            
        except Exception:
            # Ignore warmup errors, continue with benchmark
            pass
    
    print(f"{S_GREEN} Warmup completed!{E_GREEN}")


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
        return f"{seconds*1000000:.1f}�s"


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


def get_test_combinations(scope, selected_regression_types, selected_function_types):
    """Get test combinations based on user scope choice."""
    if scope == 1:
        # Test only selected combinations
        return selected_regression_types, selected_function_types
    elif scope == 2:
        # Test all regression types with selected functions
        return [1, 2, 3, 4], selected_function_types
    else:  # scope == 3
        # Test comprehensive: all regression � polynomial functions 1-7
        return [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7]


def run_performance_benchmark(X, y, selected_regression_types, selected_function_types):
    """Main benchmark function comparing all regression engines."""
    
    print(f"\n{S_BOLD}=� REGRESSION ENGINES PERFORMANCE BENCHMARK{E_BOLD}")
    print("=" * 60)
    
    # Get user settings
    num_runs, scope = get_benchmark_settings()
    
    # Get test combinations
    regression_types, function_types = get_test_combinations(
        scope, selected_regression_types, selected_function_types)
    
    # Display benchmark info
    total_combinations = len(regression_types) * len(function_types)
    print(f"\n{S_BOLD}Benchmark Configuration:{E_BOLD}")
    print(f"  • Number of runs per engine: {num_runs}")
    print(f"  • Regression types: {len(regression_types)} {regression_types}")
    print(f"  • Function types: {len(function_types)} {function_types}")
    print(f"  • Total combinations: {total_combinations}")
    print(f"  • Data points: {len(X)}")
    
    # Warmup all engines
    warmup_engines(X, y, regression_types, function_types)
    
    # Engine information
    engines = {
        1: "C++ MLPack",
        2: "NumPy", 
        3: "Numba JIT",
        4: "Pure Python"
    }
    
    print(f"\n{S_BOLD}�  Running benchmark...{E_BOLD}")
    
    # Time each engine
    results = {}
    for engine_id, engine_name in engines.items():
        print(f"  Testing {engine_name}...", end=" ", flush=True)
        
        try:
            avg_time, total_time = time_single_engine(
                engine_id, X, y, regression_types, function_types, num_runs)
            results[engine_id] = avg_time
            print(f"{S_GREEN}{E_GREEN} {format_time(avg_time)}")
            
        except Exception as e:
            results[engine_id] = None
            print(f"{S_RED}L{E_RED} Failed: {str(e)[:50]}...")
    
    # Display results
    display_benchmark_results(results, engines, num_runs, total_combinations)


def display_benchmark_results(results, engines, num_runs, total_combinations):
    """Display formatted benchmark results with speedup comparisons."""
    
    print(f"\n{S_BOLD}=� BENCHMARK RESULTS{E_BOLD}")
    print("=" * 60)
    
    # Find baseline (C++ if available, otherwise fastest)
    baseline_time = None
    baseline_engine = None
    
    if results.get(1) is not None:  # C++ available
        baseline_time = results[1]
        baseline_engine = "C++ MLPack"
    else:
        # Use fastest available engine as baseline
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            baseline_engine_id = min(valid_results, key=valid_results.get)
            baseline_time = valid_results[baseline_engine_id]
            baseline_engine = engines[baseline_engine_id]
    
    if baseline_time is None:
        print(f"{S_RED}L No successful benchmarks to display{E_RED}")
        return
    
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
        
        print(f"  {engine_name:12}: {format_time(avg_time):>8} - {speedup_str}{baseline_marker}")
    
    # Show failed engines
    failed_engines = [engines[k] for k, v in results.items() if v is None]
    if failed_engines:
        print(f"\n{S_RED}Failed engines: {', '.join(failed_engines)}{E_RED}")
    
    # Performance insights
    print(f"\n{S_BOLD}=� Performance Insights:{E_BOLD}")
    
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