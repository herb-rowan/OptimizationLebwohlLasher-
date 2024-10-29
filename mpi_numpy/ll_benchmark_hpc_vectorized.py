"""
HPC benchmarking script for Lebwohl-Lasher simulation
Tests both problem size scaling and parallel scaling
"""

from mpi4py import MPI
import numpy as np
import json
import time
import sys
from pathlib import Path
def convert_to_serializable(results):
    """Recursive function to convert all NumPy types in the dictionary to Python-native types."""
    if isinstance(results, dict):
        return {k: convert_to_serializable(v) for k, v in results.items()}
    elif isinstance(results, list):
        return [convert_to_serializable(i) for i in results]
    elif hasattr(results, 'item'):  # Numpy type
        return results.item()
    else:
        return results
def run_benchmark(size, iterations=1):
    """Run a single benchmark with given lattice size"""
    from LebwohlLasher_mpi_sequential import initdat, MC_step_vectorized
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    
    # Initialize lattice
    lattice = initdat(size)
    
    # Warmup - single step
    if rank == 0:
        print(f"Starting warmup for size {size}x{size}")
    MC_step_vectorized(lattice, 0.5, size)
    
    # Timing run
    if rank == 0:
        print(f"Starting timing run for size {size}x{size}")
    
    start_time = MPI.Wtime()
    
    for _ in range(iterations):
        MC_step_vectorized(lattice, 0.5, size)
    
    end_time = MPI.Wtime()
    runtime = end_time - start_time
    
    # Gather timing statistics
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
    """Run full benchmark suite"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Starting benchmark with {nprocs} processes")
        print(f"{'='*60}")
    
    # Define lattice sizes to test (12 sizes from 10 to 10000)
    lattice_sizes = np.logspace(1, 3, 12, dtype=int)
    
    # Results filename based on number of processes
    results_file = f'benchmark_results/results_p{nprocs:02d}.json'
    
    results = []
    
    for lattice_size in lattice_sizes:
        if rank == 0:
            print(f"\nBenchmarking lattice size: {lattice_size}x{lattice_size}")
        
        # Adjust iterations based on lattice size and process count
        # iterations = max(10, int(100 / (lattice_size * lattice_size)))
        iterations = max(1, 0)

        
        try:
            result = run_benchmark(lattice_size, iterations)
            if rank == 0 and result is not None:
                results.append(result)
                
                # Save intermediate results
                with open(results_file, 'w') as f:
                    json.dump(convert_to_serializable(results), f, indent=2)
                
                print(f"Completed size {lattice_size}x{lattice_size}:")
                print(f"Time per step: {result['time_per_step']:.3f} seconds")
                print(f"Total time: {result['total_time']:.3f} seconds")
        
        except Exception as e:
            if rank == 0:
                print(f"Error benchmarking size {lattice_size}: {e}")
                break
    
    if rank == 0:
        # Save final results
        with open(results_file, 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)
        print(f"\nBenchmark completed. Results saved to {results_file}")

if __name__ == "__main__":
    main()