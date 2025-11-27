#!/usr/bin/env python3
"""
Main application using the CUDA algorithm module
"""

import sys
import os

# Add module build directory to Python path
module_build_path = os.path.join(os.path.dirname(__file__), '..', 'module', 'build')
if os.path.exists(module_build_path):
    # Find the Python module (algorithm_python*.pyd on Windows, algorithm_python*.so on Linux)
    for root, dirs, files in os.walk(module_build_path):
        for file in files:
            if file.startswith('algorithm') and (file.endswith('.pyd') or file.endswith('.so')):
                sys.path.insert(0, root)
                # Also add the directory containing the module
                if os.path.dirname(root) not in sys.path:
                    sys.path.insert(0, os.path.dirname(root))
                break

try:
    import algorithm
    print("Successfully imported algorithm module")
except ImportError as e:
    print(f"Failed to import algorithm module: {e}")
    print("Make sure the module is built and the Python extension is available")
    sys.exit(1)


def main():
    """Main function demonstrating vector addition using CUDA"""
    
    print("=" * 50)
    print("CUDA Vector Addition Example")
    print("=" * 50)
    
    # Example 1: Small vectors
    print("\nExample 1: Adding small vectors")
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [10.0, 20.0, 30.0, 40.0, 50.0]
    
    print(f"Vector A: {a}")
    print(f"Vector B: {b}")
    
    try:
        result = algorithm.add_vectors(a, b)
        print(f"Result:   {result}")
        print(f"Expected: [11.0, 22.0, 33.0, 44.0, 55.0]")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Larger vectors
    print("\nExample 2: Adding larger vectors")
    size = 1000
    a = [float(i) for i in range(size)]
    b = [float(i * 2) for i in range(size)]
    
    print(f"Vector size: {size}")
    print(f"First 5 elements of A: {a[:5]}")
    print(f"First 5 elements of B: {b[:5]}")
    
    try:
        result = algorithm.add_vectors(a, b)
        print(f"First 5 elements of result: {result[:5]}")
        print(f"Last 5 elements of result: {result[-5:]}")
        print("Vector addition completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Error handling - different sizes
    print("\nExample 3: Error handling - different vector sizes")
    a = [1.0, 2.0, 3.0]
    b = [10.0, 20.0]
    
    print(f"Vector A (size {len(a)}): {a}")
    print(f"Vector B (size {len(b)}): {b}")
    
    try:
        result = algorithm.add_vectors(a, b)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Expected error caught: {e}")
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()

