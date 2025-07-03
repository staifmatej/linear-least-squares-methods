#!/usr/bin/env python3
"""Test visualization with Numba engine."""

import subprocess
import sys

# Test with example data and visualization
# Data: 5 (Example dataset) -> 1 (House prices)
# Engine: 2 (Numba)
# Regression: 1 (Linear)
# Function: 1 (Linear)
# Then: 1 (Visualize results on one image)
inputs = "5\n1\n2\n1\n1\n1\n7\n"

try:
    result = subprocess.run(
        [sys.executable, "main.py"],
        input=inputs,
        text=True,
        capture_output=True,
        timeout=30
    )
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.returncode != 0:
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")
    else:
        print("\n✓ Success! Visualization works with Numba engine.")
        
except subprocess.TimeoutExpired:
    print("✗ Process timed out after 30 seconds")
except Exception as e:
    print(f"✗ Error: {e}")