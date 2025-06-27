"""Performance benchmarking tool for comparing regression engine speeds."""

import time
import timeit
import numpy as np
import sys
import os
import shutil
import gc
from tabulate import tabulate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.run_regression import RegressionRun
from utils.after_regression_handler import print_press_enter_to_continue

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
    print(f"{S_YELLOW}Clearing caches...{E_YELLOW}", end=" ", flush=True)
    
    # Clear Python bytecode cache silently
    try:
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(current_dir)
        
        for root, dirs, files in os.walk(project_root):
            if '__pycache__' in dirs:
                pycache_path = os.path.join(root, '__pycache__')
                try:
                    shutil.rmtree(pycache_path)
                except Exception:
                    pass
    except Exception:
        pass
    
    # Force garbage collection silently
    try:
        gc.collect()
    except Exception:
        pass
    
    # Clear import cache for Python modules silently
    try:
        modules_to_clear = [
            'approaches.least_squares_numba',
            'approaches.least_squares_numpy', 
            'approaches.least_squares_pure'
        ]
        
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                del sys.modules[module_name]
    except Exception:
        pass
    
    print(f"{S_GREEN}Done{E_GREEN}")


def get_benchmark_settings():
    """Get user preferences for benchmark settings - simplified version."""

    while 1:
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
    
    print(f"\n{S_BOLD}Automatic benchmark configuration:{E_BOLD}")
    print("  • Testing all engines: NumPy, Numba (with & without warm up), Pure Python")
    print("  • Testing both pipeline and core performance")
    print("  • Testing polynomial functions of degrees 1-7")
    
    return num_runs


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


def time_single_engine_cold_numba(X, y, regression_types, function_types, num_runs):
    """Time Numba engine with TRUE cold start - including all JIT compilation."""
    
    def run_cold_benchmark():
        # Force fresh import and JIT compilation
        import importlib
        import sys
        
        # Remove Numba modules from cache if they exist
        modules_to_remove = []
        for module_name in sys.modules.keys():
            if 'least_squares_numba' in module_name:
                modules_to_remove.append(module_name)
        
        for module_name in modules_to_remove:
            del sys.modules[module_name]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Now run the benchmark - this will trigger JIT compilation
        runner = RegressionRun(2, regression_types, function_types)  # 2 = Numba engine
        results = runner.run_regressions(X, y)
        return results
    
    # Time including all JIT compilation overhead
    total_time = timeit.timeit(run_cold_benchmark, number=num_runs)
    avg_time = total_time / num_runs
    
    return avg_time, total_time


def format_time(seconds):
    """Format time with appropriate units."""
    if seconds >= 60.0:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    elif seconds >= 1.0:
        return f"{seconds:.3f}s"
    elif seconds >= 0.001:
        return f"{seconds*1000:.1f}ms"
    else:
        return f"{seconds*1000000:.1f}µs"


def run_comprehensive_benchmark(X, y, regression_types, function_types, num_runs):
    """Run full pipeline benchmark - complete implementation with fitting."""
    

    # Test NumPy and Pure Python (normal engines)
    pipeline_results = {}
    

    engines = {1: "NumPy", 3: "Pure Python"}
    
    for engine_id, engine_name in engines.items():
        print(f"  Testing {engine_name} full pipeline...", end=" ", flush=True)
        clear_all_caches()  # Clear for each test
        try:
            avg_time, total_time = time_single_engine(
                engine_id, X, y, regression_types, function_types, num_runs)
            if avg_time is not None:
                pipeline_results[engine_name] = avg_time
                print(f"\n{S_GREEN}{format_time(avg_time)}{E_GREEN}")
            else:
                pipeline_results[engine_name] = None
                print(f"{S_RED}Failed{E_RED}")
        except Exception as e:
            pipeline_results[engine_name] = None
            print(f"{S_RED}Failed{E_RED}")
    
    # Test Numba - COLD START (including JIT compilation time)
    print(f"\n{S_BOLD}Testing Numba COLD START (with JIT compilation):{E_BOLD}")
    print(f"  Testing Numba pipeline (COLD)...", end=" ", flush=True)
    clear_all_caches()  # Critical: clear ALL caches for true cold start
    try:
        avg_time, total_time = time_single_engine_cold_numba(
            X, y, regression_types, function_types, num_runs)
        if avg_time is not None:
            pipeline_results["Numba (Cold)"] = avg_time
            print(f"\n{S_GREEN}{format_time(avg_time)}{E_GREEN}")
        else:
            pipeline_results["Numba (Cold)"] = None
            print(f"{S_RED}Failed{E_RED}")
    except Exception as e:
        pipeline_results["Numba (Cold)"] = None
        print(f"{S_RED}Failed{E_RED}")
    
    # Test Numba - WARM START (after JIT compilation)  
    print(f"\n{S_BOLD}Testing Numba WARM START (JIT pre-compiled):{E_BOLD}")
    print(f"  Testing Numba pipeline (WARM)...\n", flush=True)
    # Don't clear caches - use already compiled Numba functions
    try:
        avg_time, total_time = time_single_engine(
            2, X, y, regression_types, function_types, num_runs)
        if avg_time is not None:
            pipeline_results["Numba (Warm)"] = avg_time
            print(f"{S_GREEN}{format_time(avg_time)}{E_GREEN}")
        else:
            pipeline_results["Numba (Warm)"] = None
            print(f"{S_RED}Failed{E_RED}")
    except Exception as e:
        pipeline_results["Numba (Warm)"] = None
        print(f"{S_RED}Failed{E_RED}")
    
    return pipeline_results


def run_performance_benchmark(X, y, selected_regression_types, selected_function_types):
    """Main benchmark function - automatically tests all scenarios."""
    
    print("\n═══════ COMPREHENSIVE REGRESSION ENGINES BENCHMARK ═══════\n")

    # Get only number of runs from user
    num_runs = get_benchmark_settings()
    
    # Fixed configuration
    regression_types = [1, 2, 3, 4]  # Linear, Ridge, Lasso, ElasticNet
    function_types = [1, 2, 3, 4, 5, 6, 7]  # Polynomial functions 1-7
    total_combinations = len(regression_types) * len(function_types)
    
    print(f"\n{S_BOLD}Running comprehensive benchmark with:{E_BOLD}")
    print(f"  • Number of runs per test: {num_runs}")
    print(f"  • Regression types: {len(regression_types)} (Linear, Ridge, Lasso, ElasticNet)")
    print(f"  • Function types: {len(function_types)} (Polynomial degrees 1-7)")
    print(f"  • Total combinations: {total_combinations}")
    print(f"  • Data points: {len(X)}")
    print(f"  • Engines: NumPy, Numba (cold + warm), Pure Python\n")
    
    # Run comprehensive benchmark
    pipeline_results = run_comprehensive_benchmark(X, y, regression_types, function_types, num_runs)
    
    # Display final comparison using tabulate
    display_pipeline_results_table(pipeline_results, num_runs, total_combinations)


def display_pipeline_results_table(pipeline_results, num_runs, total_combinations):
    """Display pipeline results using tabulate for nice formatting."""
    
    print(f"\n{S_BOLD}══════════ FULL IMPLEMENTATION BENCHMARK RESULTS ══════════{E_BOLD}")

    # Prepare data for pipeline table  
    pipeline_table_data = []
    valid_pipeline = [(k, v) for k, v in pipeline_results.items() if v is not None]
    
    if valid_pipeline:
        # Sort by performance (fastest first)
        valid_pipeline.sort(key=lambda x: x[1])
        fastest_pipeline_time = valid_pipeline[0][1]
        
        print(f"\n{S_BOLD}COMPLETE REGRESSION PIPELINE PERFORMANCE{E_BOLD}")
        print(f"Full implementation with fitting, transformation, and prediction")
        print(f"Operations: {total_combinations} combinations | Runs: {num_runs} each\n")
        
        for engine_name, avg_time in valid_pipeline:
            speedup = avg_time / fastest_pipeline_time
            speedup_str = f"{speedup:.2f}x" if speedup > 1 else "baseline"
            pipeline_table_data.append([
                engine_name,
                format_time(avg_time),
                speedup_str
            ])
        
        print(tabulate(pipeline_table_data, 
                      headers=["Engine", "Average Time", "vs Fastest"],
                      tablefmt="rounded_grid",
                      stralign="left"))

    print("\n═══════════════════════════════════════════════════════════")

    print_press_enter_to_continue()