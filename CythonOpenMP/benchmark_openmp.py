import numpy as np
import time
import matplotlib.pyplot as plt
from LebwohlLasher import main as main_original
from LebwohlLasher_openmp import main as main_openmp

def run_benchmark(size, steps, temp, thread_counts):
    # Run original version
    start = time.time()
    main_original("Original", steps, size, temp, 0)
    original_time = time.time() - start
    print(f"Original version time: {original_time:.3f} seconds")
    
    # Run OpenMP version with different thread counts
    openmp_times = []
    for threads in thread_counts:
        print(f"\nRunning with {threads} threads...")
        start = time.time()
        main_openmp("OpenMP", steps, size, temp, 0, threads)
        end = time.time()
        openmp_time = end - start
        openmp_times.append(openmp_time)
        print(f"OpenMP version ({threads} threads) time: {openmp_time:.3f} seconds")
        print(f"Speedup: {original_time/openmp_time:.2f}x")
    
    return original_time, openmp_times

def plot_results(size, thread_counts, original_time, openmp_times):
    plt.figure(figsize=(10, 6))
    plt.plot(thread_counts, [original_time] * len(thread_counts), 'r--', label='Original')
    plt.plot(thread_counts, openmp_times, 'b-o', label='OpenMP')
    plt.xlabel('Number of Threads')
    plt.ylabel('Execution Time (s)')
    plt.title(f'Performance Comparison for {size}x{size} Lattice')
    plt.legend()
    plt.grid(True)
    plt.savefig('openmp_performance.png')
    plt.show()
    
    # Plot speedup
    plt.figure(figsize=(10, 6))
    speedups = [original_time/t for t in openmp_times]
    plt.plot(thread_counts, speedups, 'g-o')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.title(f'Speedup vs Number of Threads for {size}x{size} Lattice')
    plt.grid(True)
    plt.savefig('openmp_speedup.png')
    plt.show()

if __name__ == "__main__":
    # Test parameters
    size = 1000  # Lattice size
    steps = 100  # Monte Carlo steps
    temp = 0.5  # Temperature
    thread_counts = [1, 2, 4,6, 8]  # Number of threads to test
    
    # Run benchmarks
    original_time, openmp_times = run_benchmark(size, steps, temp, thread_counts)
    
    # Plot results
    plot_results(size, thread_counts, original_time, openmp_times)
    
    # Save numerical results
    with open('benchmark_results.txt', 'w') as f:
        f.write(f"Lattice size: {size}x{size}\n")
        f.write(f"Monte Carlo steps: {steps}\n")
        f.write(f"Temperature: {temp}\n\n")
        f.write(f"Original version time: {original_time:.3f}s\n\n")
        f.write("OpenMP results:\n")
        for threads, time in zip(thread_counts, openmp_times):
            speedup = original_time/time
            f.write(f"{threads} threads: {time:.3f}s (Speedup: {speedup:.2f}x)\n")