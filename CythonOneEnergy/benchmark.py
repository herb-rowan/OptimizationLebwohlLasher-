import numpy as np
import time
from LebwohlLasher import one_energy as one_energy_original
from LebwohlLasher_cython_test import one_energy_cython

def benchmark_energy_calculation(nmax, n_tests=1000000):
    # Create test array
    arr = np.random.random((nmax, nmax)) * 2.0 * np.pi
    
    # Test points
    test_points = [(np.random.randint(1, nmax-1), np.random.randint(1, nmax-1)) 
                   for _ in range(n_tests)]
    
    # Time original version
    start = time.time()
    for ix, iy in test_points:
        one_energy_original(arr, ix, iy, nmax)
    python_time = time.time() - start
    
    # Time Cython version
    start = time.time()
    for ix, iy in test_points:
        one_energy_cython(arr, ix, iy, nmax)
    cython_time = time.time() - start
    
    return python_time, cython_time

if __name__ == "__main__":
    sizes = [10, 20, 50, 100, 200, 500, 1000]
    results = []
    
    print("Lattice Size | Python Time (s) | Cython Time (s) | Speedup")
    print("-" * 60)
    
    for size in sizes:
        py_time, cy_time = benchmark_energy_calculation(size)
        speedup = py_time / cy_time
        results.append((size, py_time, cy_time, speedup))
        print(f"{size:^11d} | {py_time:^13.6f} | {cy_time:^13.6f} | {speedup:^7.2f}x")