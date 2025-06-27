#!/usr/bin/env python3
"""
Real-world benchmark without warmup cheating.
Tests different scenarios: cold start, repeated calls, batch processing.
"""

import time
import numpy as np
import subprocess
import sys
import os
from utils.run_regression import RegressionRun

def cold_start_benchmark():
    """Test cold start performance - no warmup, fresh Python process each time."""
    print("🚀 COLD START BENCHMARK (No Warmup Cheating)")
    print("=" * 60)
    
    # Generate test data
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(X.flatten()) + 0.1 * np.random.randn(100)
    
    # Save test data
    np.savez('/tmp/test_data.npz', X=X, y=y)
    
    engines = {
        1: "C++ MLPack",
        2: "NumPy", 
        3: "Numba JIT",
        4: "Pure Python"
    }
    
    regression_types = [1, 2]  # Linear, Ridge
    function_types = [1, 2, 3]  # Linear, Quadratic, Cubic
    
    results = {}
    
    for engine_id, engine_name in engines.items():
        print(f"\nTesting {engine_name} (Cold Start)...")
        
        # Create fresh Python script for each engine
        script_content = f'''
import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.run_regression import RegressionRun

# Load test data
data = np.load('/tmp/test_data.npz')
X, y = data['X'], data['y']

# Time the actual work (including first-time compilation for Numba)
start_time = time.time()

runner = RegressionRun({engine_id}, {regression_types}, {function_types})
results = runner.run_regressions(X, y)

end_time = time.time()
total_time = end_time - start_time

print(f"RESULT: {{total_time*1000:.1f}}ms")
'''
        
        script_path = f'/tmp/test_engine_{engine_id}.py'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        try:
            # Run in fresh Python process (no warmup possible)
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Extract timing from output
                for line in result.stdout.split('\n'):
                    if line.startswith('RESULT:'):
                        time_ms = float(line.split(':')[1].replace('ms', '').strip())
                        results[engine_id] = time_ms
                        print(f"  ✅ {time_ms:.1f}ms")
                        break
                else:
                    print(f"  ❌ Failed to parse result")
                    results[engine_id] = None
            else:
                print(f"  ❌ Failed: {result.stderr[:100]}...")
                results[engine_id] = None
                
        except subprocess.TimeoutExpired:
            print(f"  ❌ Timeout")
            results[engine_id] = None
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[engine_id] = None
    
    # Display results
    print(f"\n📊 COLD START RESULTS")
    print("=" * 60)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    if not valid_results:
        print("❌ No successful results")
        return
    
    # Sort by performance
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1])
    baseline_time = sorted_results[0][1]
    
    print("Ranking (Fresh Process, No Warmup):")
    for i, (engine_id, time_ms) in enumerate(sorted_results, 1):
        engine_name = engines[engine_id]
        speedup = time_ms / baseline_time
        print(f"{i}. {engine_name:20}: {time_ms:6.1f}ms ({speedup:.2f}x)")
    
    return valid_results

def repeated_calls_benchmark():
    """Test performance with repeated calls (realistic for long-running apps)."""
    print("\n🚀 REPEATED CALLS BENCHMARK (Long-running App Simulation)")
    print("=" * 60)
    
    # Generate test data
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(X.flatten()) + 0.1 * np.random.randn(100)
    
    engines = {
        1: "C++ MLPack",
        2: "NumPy", 
        3: "Numba JIT",
        4: "Pure Python"
    }
    
    regression_types = [1, 2]  # Linear, Ridge
    function_types = [1, 2, 3]  # Linear, Quadratic, Cubic
    
    results = {}
    num_calls = 10  # Simulate 10 regression calls
    
    for engine_id, engine_name in engines.items():
        print(f"\nTesting {engine_name} ({num_calls} repeated calls)...")
        
        try:
            runner = RegressionRun(engine_id, regression_types, function_types)
            
            # First call (includes warmup for Numba)
            print("  First call (includes any warmup)...", end="")
            start_time = time.time()
            runner.run_regressions(X, y)
            first_call_time = time.time() - start_time
            print(f" {first_call_time*1000:.1f}ms")
            
            # Subsequent calls (hot performance)
            print(f"  Next {num_calls-1} calls (hot performance)...", end="")
            times = []
            for i in range(num_calls - 1):
                start_time = time.time()
                runner.run_regressions(X, y)
                times.append(time.time() - start_time)
            
            avg_hot_time = np.mean(times) * 1000
            print(f" avg {avg_hot_time:.1f}ms")
            
            # Total time for realistic use
            total_time = first_call_time + sum(times)
            avg_overall = total_time / num_calls * 1000
            
            results[engine_id] = {
                'first_call': first_call_time * 1000,
                'hot_avg': avg_hot_time,
                'overall_avg': avg_overall
            }
            
            print(f"  📊 Overall average: {avg_overall:.1f}ms per call")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[engine_id] = None
    
    # Display results
    print(f"\n📊 REPEATED CALLS RESULTS")
    print("=" * 60)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    if not valid_results:
        print("❌ No successful results")
        return
    
    print("Performance breakdown:")
    print("Engine               First Call    Hot Avg    Overall Avg")
    print("-" * 60)
    
    for engine_id, timings in valid_results.items():
        engine_name = engines[engine_id]
        print(f"{engine_name:18} {timings['first_call']:8.1f}ms {timings['hot_avg']:8.1f}ms {timings['overall_avg']:10.1f}ms")
    
    # Ranking by overall average (most realistic)
    sorted_overall = sorted(valid_results.items(), key=lambda x: x[1]['overall_avg'])
    baseline_overall = sorted_overall[0][1]['overall_avg']
    
    print(f"\nRanking by Overall Average ({num_calls} calls):")
    for i, (engine_id, timings) in enumerate(sorted_overall, 1):
        engine_name = engines[engine_id]
        speedup = timings['overall_avg'] / baseline_overall
        print(f"{i}. {engine_name:20}: {timings['overall_avg']:6.1f}ms ({speedup:.2f}x)")
    
    return valid_results

def batch_processing_benchmark():
    """Test batch processing scenario (many datasets)."""
    print("\n🚀 BATCH PROCESSING BENCHMARK (Multiple Datasets)")
    print("=" * 60)
    
    engines = {
        1: "C++ MLPack",
        2: "NumPy", 
        3: "Numba JIT", 
        4: "Pure Python"
    }
    
    regression_types = [1]  # Just Linear for speed
    function_types = [1, 2]  # Linear, Quadratic
    
    # Generate 5 different datasets
    datasets = []
    for i in range(5):
        X = np.linspace(0, 10, 50).reshape(-1, 1) + np.random.normal(0, 0.1, (50, 1))
        y = np.sin(X.flatten()) + 0.1 * np.random.randn(50)
        datasets.append((X, y))
    
    results = {}
    
    for engine_id, engine_name in engines.items():
        print(f"\nTesting {engine_name} (5 datasets)...")
        
        try:
            # Time processing all datasets
            start_time = time.time()
            
            for i, (X, y) in enumerate(datasets):
                runner = RegressionRun(engine_id, regression_types, function_types)
                runner.run_regressions(X, y)
                print(f"  Dataset {i+1}/5 completed", end="\r")
            
            total_time = time.time() - start_time
            avg_per_dataset = total_time / len(datasets) * 1000
            
            results[engine_id] = {
                'total_time': total_time * 1000,
                'avg_per_dataset': avg_per_dataset
            }
            
            print(f"  ✅ Total: {total_time*1000:.1f}ms, Avg: {avg_per_dataset:.1f}ms/dataset")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[engine_id] = None
    
    # Display results
    print(f"\n📊 BATCH PROCESSING RESULTS")
    print("=" * 60)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    if not valid_results:
        print("❌ No successful results")
        return
        
    sorted_batch = sorted(valid_results.items(), key=lambda x: x[1]['avg_per_dataset'])
    baseline_batch = sorted_batch[0][1]['avg_per_dataset']
    
    print("Ranking by Average Time per Dataset:")
    for i, (engine_id, timings) in enumerate(sorted_batch, 1):
        engine_name = engines[engine_id]
        speedup = timings['avg_per_dataset'] / baseline_batch
        print(f"{i}. {engine_name:20}: {timings['avg_per_dataset']:6.1f}ms ({speedup:.2f}x)")
    
    return valid_results

def main():
    """Run comprehensive real-world benchmark."""
    print("🔍 REAL-WORLD PERFORMANCE ANALYSIS")
    print("Testing different scenarios without warmup cheating")
    print("=" * 70)
    
    # Test 1: Cold start (most honest)
    cold_results = cold_start_benchmark()
    
    # Test 2: Repeated calls (long-running app)
    repeated_results = repeated_calls_benchmark()
    
    # Test 3: Batch processing (multiple datasets)
    batch_results = batch_processing_benchmark()
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    print("\n• COLD START (Script/Batch Jobs): Most honest test")
    print("• REPEATED CALLS (Long-running Apps): Realistic for servers/daemons") 
    print("• BATCH PROCESSING (Multiple Datasets): Real data science workflow")
    
    print(f"\nConclusion:")
    print(f"- For one-time scripts: Cold start performance matters most")
    print(f"- For interactive/server use: Repeated calls performance matters")
    print(f"- Numba's JIT compilation cost is front-loaded")

if __name__ == "__main__":
    main()