"""
HPC benchmarking script with adaptive iterations for large lattices
"""

from mpi4py import MPI
import numpy as np
import json
import time
import sys
from pathlib import Path

def get_iterations(size):
    """Determine number of iterations based on lattice size"""
    if size <= 50:
        return 20
    elif size <= 100:
        return 15
    elif size <= 500:
        return 10
    elif size <= 1000:
        return 5
    else:
        return 3  # Minimum iterations for very large lattices

def run_benchmark(size):
    """Run benchmark with iterations adapted to size"""
    from LebwohlLasher_mpi import initdat, MC_step
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    
    iterations = get_iterations(size)
    
    if rank == 0:
        print(f"Testing {size}x{size} with {iterations} iterations on {nprocs} processes")
    
    # Initialize lattice
    lattice = initdat(size)
    
    start_time = MPI.Wtime()
    
    for _ in range(iterations):
        MC_step(lattice, 0.5, size)
    
    end_time = MPI.Wtime()
    runtime = end_time - start_time
    
    all_runtimes = comm.gather(runtime, root=0)
    
    if rank == 0:
        avg_runtime = np.mean(all_runtimes)
        return {
            'size': size,
            'processes': nprocs,
            'iterations': iterations,
            'total_time': avg_runtime,
            'time_per_step': avg_runtime / iterations
        }
    return None

def main():
    """Run benchmark suite with adaptive iterations"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    
    # Test a wide range of sizes
    lattice_sizes = [50, 100, 200, 500, 1000, 2000]
    
    results_file = f'benchmark_results/results_p{nprocs:02d}.json'
    results = []
    
    for lattice_size in lattice_sizes:
        try:
            result = run_benchmark(lattice_size)
            if rank == 0 and result is not None:
                results.append(result)
                print(f"\nCompleted {lattice_size}x{lattice_size}:")
                print(f"Total time: {result['total_time']:.3f} seconds")
                print(f"Time per step: {result['time_per_step']:.3f} seconds")
                
                # Early warning if times are getting too long
                if result['time_per_step'] > 60:  # More than 1 minute per step
                    print("WARNING: Execution times are getting very long!")
                    break
                
                # Save after each size in case we need to stop early
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
        
        except Exception as e:
            if rank == 0:
                print(f"Error benchmarking size {lattice_size}: {e}")
            break
    
    if rank == 0:
        print(f"\nBenchmark completed. Results saved to {results_file}")

if __name__ == "__main__":
    main()