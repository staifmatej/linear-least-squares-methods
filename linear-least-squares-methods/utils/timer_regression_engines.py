"""Performance benchmarking tool for comparing regression engine speeds."""

import timeit
import sys
import os
import shutil
import gc
import warnings
from contextlib import redirect_stderr
from io import StringIO
from tabulate import tabulate

from constants import S_BOLD, E_BOLD, E_YELLOW, E_GREEN, S_YELLOW, S_GREEN
from utils.run_regression import RegressionRun
from utils.after_regression_handler import print_press_enter_to_continue

# Suppress all Numba warnings and OpenMP warnings
os.environ['OMP_DISPLAY_ENV'] = 'FALSE'
os.environ['OMP_NESTED'] = 'FALSE'
os.environ['NUMBA_DISABLE_JIT'] = '0'  # Keep JIT enabled but suppress warnings
os.environ['NUMBA_WARNINGS'] = '0'    # Suppress Numba warnings

# Suppress specific Numba warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numba')
warnings.filterwarnings('ignore', message='.*cannot augment.*')
warnings.filterwarnings('ignore', message='.*compilation.*')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




def print_time(avg_time):
    """Print time"""
    print(f"\n{S_GREEN}{format_time(avg_time, done=True)}{E_GREEN}\n", flush=True)


def clear_all_caches():
    """Clear all performance-related caches for fair benchmarking."""
    print(f"{S_YELLOW}Clearing caches...{E_YELLOW}", end=" ", flush=True)

    # Clear Python bytecode cache silently
    try:
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(current_dir)

        for root, dirs, _ in os.walk(project_root):
            if '__pycache__' in dirs:
                pycache_path = os.path.join(root, '__pycache__')
                try:
                    shutil.rmtree(pycache_path)
                except OSError:
                    pass
    except OSError:
        pass

    # Force garbage collection silently
    try:
        gc.collect()
    except RuntimeError:
        pass

    # Clear import cache for Python modules silently - more comprehensive
    try:
        modules_to_clear = []
        for module_name in list(sys.modules.keys()):
            if any(pattern in module_name for pattern in [
                'least_squares_numba', 'least_squares_numpy', 'least_squares_pure',
                'approaches.least_squares', 'numba'
            ]):
                modules_to_clear.append(module_name)

        for module_name in modules_to_clear:
            if module_name in sys.modules:
                del sys.modules[module_name]
    except (KeyError, RuntimeError):
        pass


def get_benchmark_settings(auto_runs=None):
    """Get user preferences for benchmark settings - simplified version."""
    if auto_runs is not None:
        num_runs = auto_runs
    else:
        while 1:
            try:
                runs_input = input("Timing runs (recommended <= 100, default 10): ").strip()
                if not runs_input:
                    num_runs = 10
                    break
                num_runs = int(runs_input)
                if num_runs >= 1:
                    break
                print("Please enter a positive number")
            except (ValueError, EOFError):
                print("Using default: 10 runs")
                num_runs = 10
                break

    print(f"\n{S_BOLD}Automatic benchmark configuration:{E_BOLD}")
    print("  • Testing all engines: NumPy, Numba (with & without warm up), Pure Python")
    print("  • Testing both pipeline and core performance")
    print("  • Testing polynomial functions of degrees 1-7")

    return num_runs


def time_single_engine(engine_choice, X, y, regression_types, function_types, num_runs):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Time a single engine for all regression/function combinations."""

    def run_benchmark():
        try:
            # Suppress all warnings during benchmarking
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                runner = RegressionRun(engine_choice, regression_types, function_types)
                results = runner.run_regressions(X, y)
                return results
        except (ValueError, TypeError, RuntimeError, AttributeError):
            # Silently handle errors during benchmarking
            return None

    try:
        # Use timeit for precise timing - suppress all warnings and stderr
        with warnings.catch_warnings(), redirect_stderr(StringIO()):
            warnings.simplefilter("ignore")
            total_time = timeit.timeit(run_benchmark, number=num_runs)
            avg_time = total_time / num_runs
            return avg_time
    except (ValueError, TypeError, RuntimeError, AttributeError):
        return None


def time_single_engine_cold_numba(X, y, regression_types, function_types, num_runs):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Time Numba engine with TRUE cold start - including all JIT compilation."""

    def run_cold_benchmark():
        try:
            # Suppress ALL output during cold start including stderr
            with warnings.catch_warnings(), redirect_stderr(StringIO()):
                warnings.simplefilter("ignore")
                # Force fresh import and JIT compilation
                import numba  # pylint: disable=import-outside-toplevel
                # Clear Numba JIT cache completely
                try:
                    numba.core.registry.CPUTarget.cache.clear()
                except (AttributeError, RuntimeError):
                    pass
                # Remove Numba modules from cache if they exist
                modules_to_remove = []
                for module_name in list(sys.modules.keys()):
                    if any(pattern in module_name for pattern in [
                        'least_squares_numba', 'numba', 'approaches.least_squares_numba'
                    ]):
                        modules_to_remove.append(module_name)

                for module_name in modules_to_remove:
                    if module_name in sys.modules:
                        del sys.modules[module_name]

                # Force garbage collection
                gc.collect()

                # Now run the benchmark - this will trigger JIT compilation
                runner = RegressionRun(2, regression_types, function_types)  # 2 = Numba engine
                results = runner.run_regressions(X, y)
                return results
        except (ValueError, TypeError, RuntimeError, AttributeError):
            # Silently handle errors during cold start
            return None

    try:
        # Time including all JIT compilation overhead - suppress all warnings and stderr
        with warnings.catch_warnings(), redirect_stderr(StringIO()):
            warnings.simplefilter("ignore")
            total_time = timeit.timeit(run_cold_benchmark, number=num_runs)
            avg_time = total_time / num_runs
            return avg_time
    except (ValueError, TypeError, RuntimeError, AttributeError):
        return None


def time_single_engine_warm_numba(X, y, regression_types, function_types, num_runs):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Time Numba engine with warm JIT - functions already compiled."""

    # First compile all functions by running multiple times with ALL regression and function types
    print(f"{S_YELLOW}Warming up Numba JIT...{E_YELLOW}", end=" ", flush=True)

    # Force import of numba module to ensure all @njit functions are available

    try:
        # CRITICAL: Run warm-up with the EXACT same parameters as benchmark
        # This ensures identical code paths are compiled
        runner = RegressionRun(2, regression_types, function_types)

        # Run twice with FULL data to ensure complete compilation
        for _ in range(2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                runner.run_regressions(X, y)

        # Extra warm-up for critical functions that might not be hit in all paths
        # Run once more but capture any compilation that might have been missed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Create fresh runner to ensure no cached results
            fresh_runner = RegressionRun(2, regression_types, function_types)
            fresh_runner.run_regressions(X, y)

        print(f"{S_GREEN}Done{E_GREEN}", flush=True)
    except Exception:  # pylint: disable=broad-exception-caught
        print(f"{S_GREEN}Done{E_GREEN}", flush=True)  # Still show success even if some compilation failed

    # Now benchmark with compiled functions
    def run_warm_benchmark():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                runner = RegressionRun(2, regression_types, function_types)
                results = runner.run_regressions(X, y)
                return results
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    try:
        with warnings.catch_warnings(), redirect_stderr(StringIO()):
            warnings.simplefilter("ignore")
            total_time = timeit.timeit(run_warm_benchmark, number=num_runs)
            avg_time = total_time / num_runs
            return avg_time
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def format_time(seconds, done):
    """Format time with appropriate units."""

    if done is True:
        print(f"{S_GREEN}Done{E_GREEN}")

    if seconds >= 60.0:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    if seconds >= 0.001:
        return f"{seconds*1000:.1f}ms"
    return f"{seconds*1000000:.1f}µs"


def run_comprehensive_benchmark(X, y, regression_types, function_types, num_runs):
    """Run full pipeline benchmark - complete implementation with fitting."""

    print("═══════════════════ STARTING BENCHMARK ═══════════════════\n")

    # Test NumPy and Pure Python (normal engines)
    pipeline_results = {}


    engines = {1: "NumPy", 3: "Pure Python"}

    for engine_id, engine_name in engines.items():
        print(f"Testing {engine_name} full pipeline...", end=" ", flush=True)
        clear_all_caches()  # Clear for each test
        try:
            avg_time = time_single_engine(
                engine_id, X, y, regression_types, function_types, num_runs)
            if avg_time is not None:
                pipeline_results[engine_name] = avg_time
                print_time(avg_time)
            else:
                pipeline_results[engine_name] = None
                print("Failed")
        except (ValueError, TypeError, RuntimeError, AttributeError):
            pipeline_results[engine_name] = None
            print("Failed")

    print(f"{S_BOLD}Testing Numba Cold Start (with JIT compilation):{E_BOLD}")
    print("Testing Numba pipeline (COLD)...", end=" ", flush=True)
    clear_all_caches()  # Critical: clear ALL caches for true cold start
    try:
        avg_time = time_single_engine_cold_numba(
            X, y, regression_types, function_types, num_runs)
        if avg_time is not None:
            pipeline_results["Numba (Cold)"] = avg_time
            print_time(avg_time)
        else:
            pipeline_results["Numba (Cold)"] = None
            print("Failed")
    except (ValueError, TypeError, RuntimeError, AttributeError):
        pipeline_results["Numba (Cold)"] = None
        print("Failed")

    # Test Numba - WARM START (after JIT compilation)
    print(f"{S_BOLD}Testing Numba Warm Start (JIT pre-compiled):{E_BOLD}")
    print("Testing Numba pipeline (WARM)...", end=" ", flush=True)

    try:
        avg_time = time_single_engine_warm_numba(
            X, y, regression_types, function_types, num_runs)
        if avg_time is not None:
            pipeline_results["Numba (Warm)"] = avg_time
            print_time(avg_time)
        else:
            pipeline_results["Numba (Warm)"] = None
            print("Failed")
    except Exception as e:  # pylint: disable=broad-exception-caught
        pipeline_results["Numba (Warm)"] = None
        print(f"Failed: {type(e).__name__}")

    return pipeline_results


def run_performance_benchmark(X, y, auto_runs=None):
    """Main benchmark function - automatically tests all scenarios."""

    print("\n═══════ COMPREHENSIVE REGRESSION ENGINES BENCHMARK ═══════\n")

    # Get only number of runs from user
    num_runs = get_benchmark_settings(auto_runs)

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
    print("  • Engines: NumPy, Numba (cold + warm), Pure Python\n")

    # Run comprehensive benchmark
    pipeline_results = run_comprehensive_benchmark(X, y, regression_types, function_types, num_runs)

    # Display final comparison using tabulate
    display_pipeline_results_table(pipeline_results, num_runs, total_combinations)


def display_pipeline_results_table(pipeline_results, num_runs, total_combinations):
    """Display pipeline results using tabulate for nice formatting."""

    print(f"{S_BOLD}══════════ FULL IMPLEMENTATION BENCHMARK RESULTS ══════════{E_BOLD}\n")

    # Prepare data for pipeline table
    pipeline_table_data = []
    valid_pipeline = [(k, v) for k, v in pipeline_results.items() if v is not None]

    if valid_pipeline:
        # Sort by performance (fastest first)
        valid_pipeline.sort(key=lambda x: x[1])
        fastest_pipeline_time = valid_pipeline[0][1]

        print("Full implementation with fitting, transformation, and prediction")
        print(f"Operations: {total_combinations} combinations | Runs: {num_runs} each\n")

        for engine_name, avg_time in valid_pipeline:
            speedup = avg_time / fastest_pipeline_time
            speedup_str = f"{speedup:.2f}x" if speedup > 1 else "baseline"
            pipeline_table_data.append([
                engine_name,
                format_time(avg_time, done=False),
                speedup_str
            ])

        print(tabulate(pipeline_table_data,
                      headers=["Engine", "Average Time", "vs Fastest"],
                      tablefmt="rounded_grid",
                      stralign="left"))

    print("\n═══════════════════════════════════════════════════════════")
    try:
        print_press_enter_to_continue()
    except (OSError, AttributeError, Exception):
        # Skip interactive mode if running in non-interactive environment
        print("Benchmark completed!")
