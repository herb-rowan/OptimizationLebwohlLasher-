import time
import numpy as np
from ll_parallel import run_simulation

def run_timing_test(n_threads, n_repeats=3):
    times = []
    for _ in range(n_repeats):
        initial = time.time()
        run_simulation(n_threads)
        final = time.time()
        times.append(final - initial)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time

def main():
    thread_counts = [2, 4, 8,16]
    print("\nRunning parallel Lebwohl-Lasher simulation")
    print("Parameters: 500x500 lattice, 100 MC steps, T* = 0.5")
    print("\nResults (averaged over 3 runs):")
    print("-" * 50)
    print(f"{'Threads':^10} {'Time (s)':^15} {'Std Dev':^15}")
    print("-" * 50)
    
    for threads in thread_counts:
        avg_time, std_time = run_timing_test(threads)
        print(f"{threads:^10d} {avg_time:^15.6f} {std_time:^15.6f}")
    
    print("-" * 50)

if __name__ == '__main__':
    main()