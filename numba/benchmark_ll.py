"""
Performance evaluation script for comparing original and Numba-optimized
Lebwohl-Lasher implementations
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import LebwohlLasher as ll_original
import LebwohlLasherNumba as ll_numba  # Save the Numba version as this filename

def run_benchmark(implementation, nsteps, size, temp):
    """Run a single benchmark"""
    lattice = implementation.initdat(size)
    energy = np.zeros(nsteps+1)
    ratio = np.zeros(nsteps+1)
    order = np.zeros(nsteps+1)
    
    # Set initial values
    energy[0] = implementation.all_energy(lattice, size)
    ratio[0] = 0.5
    order[0] = (implementation.get_order(lattice, size) if implementation == ll_original 
                else implementation.get_order_tensor(lattice, size))
    
    # Time the main loop
    start = time.time()
    for it in range(1, nsteps+1):
        ratio[it] = implementation.MC_step(lattice, temp, size)
        energy[it] = implementation.all_energy(lattice, size)
        order[it] = (implementation.get_order(lattice, size) if implementation == ll_original 
                    else implementation.get_order_tensor(lattice, size))
    end = time.time()
    
    return end - start, energy[-1], order[-1]

def compare_performance():
    """Compare performance between original and Numba versions"""
    # Test parameters
    temperatures = [0.5]
    sizes = [20, 30, 40, 50, 60]
    nsteps = 50
    repeats = 3
    
    # Storage for results
    original_times = {size: [] for size in sizes}
    numba_times = {size: [] for size in sizes}
    
    # Run benchmarks
    for size in sizes:
        print(f"\nTesting lattice size {size}x{size}")
        for temp in temperatures:
            print(f"Temperature: {temp}")
            for r in range(repeats):
                print(f"  Repeat {r+1}/{repeats}")
                
                # Run original version
                time_orig, energy_orig, order_orig = run_benchmark(ll_original, nsteps, size, temp)
                original_times[size].append(time_orig)
                
                # Run Numba version
                time_numba, energy_numba, order_numba = run_benchmark(ll_numba, nsteps, size, temp)
                numba_times[size].append(time_numba)
                
                # Check results are similar (within tolerance)
                if abs(energy_orig - energy_numba)/abs(energy_orig) > 0.1:
                    print("Warning: Large energy difference detected!")
                if abs(order_orig - order_numba) > 0.1:
                    print("Warning: Large order parameter difference detected!")
    
    # Calculate average times and speedups
    avg_original = [np.mean(original_times[size]) for size in sizes]
    avg_numba = [np.mean(numba_times[size]) for size in sizes]
    speedups = [orig/numba for orig, numba in zip(avg_original, avg_numba)]
    
    # Plot results
    plt.figure(figsize=(10, 5))
    
    # Runtime comparison
    plt.subplot(1, 2, 1)
    plt.plot(sizes, avg_original, 'o-', label='Original')
    plt.plot(sizes, avg_numba, 's-', label='Numba')
    plt.xlabel('Lattice Size')
    plt.ylabel('Runtime (s)')
    plt.title('Runtime vs Lattice Size')
    plt.legend()
    plt.grid(True)
    
    # Speedup
    plt.subplot(1, 2, 2)
    plt.plot(sizes, speedups, 'D-')
    plt.xlabel('Lattice Size')
    plt.ylabel('Speedup Factor')
    plt.title('Numba Speedup vs Lattice Size')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.show()
    
    # Print summary
    print("\nPerformance Summary:")
    print("==================")
    print("Lattice Size | Original (s) | Numba (s) | Speedup")
    print("------------+--------------+-----------+--------")
    for size, orig, numba, speedup in zip(sizes, avg_original, avg_numba, speedups):
        print(f"{size:^11d} | {orig:^12.3f} | {numba:^9.3f} | {speedup:^7.2f}")

if __name__ == '__main__':
    compare_performance()