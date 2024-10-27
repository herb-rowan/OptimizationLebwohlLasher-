
# """
# Performance evaluation script for MPI Lebwohl-Lasher with larger problem sizes.
# """

# import subprocess
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import sys
# from pathlib import Path

# def run_simulation(processes, size, iterations=100, temp=0.5, timeout=300):
#     """Run a single simulation with given parameters."""
#     cmd = [
#         'mpiexec',
#         '-n', str(processes),
#         sys.executable,
#         'LebwohlLasher_mpi.py',
#         str(iterations),
#         str(size),
#         str(temp),
#         '0'
#     ]
    
#     print(f"\nRunning simulation:")
#     print(f"Processes: {processes}")
#     print(f"Size: {size}x{size}")
#     print(f"Iterations: {iterations}")
    
#     try:
#         start_time = time.time()
#         result = subprocess.run(cmd, 
#                               capture_output=True,
#                               text=True,
#                               timeout=timeout)
        
#         if result.returncode != 0:
#             print("Error in simulation:")
#             print(result.stderr)
#             return None
            
#         # Extract runtime
#         for line in result.stdout.split('\n'):
#             if "Time:" in line:
#                 runtime = float(line.split("Time:")[1].split()[0])
#                 return runtime
                
#     except subprocess.TimeoutExpired:
#         print(f"Simulation timed out after {timeout} seconds")
#     except Exception as e:
#         print(f"Error: {e}")
    
#     return None

# def run_scaling_tests():
#     """Run comprehensive scaling tests."""
#     # Test parameters
#     sizes = [50, 100, 200]  # Larger sizes
#     process_counts = [1, 2, 4]  # Different numbers of processes
#     iterations = 50  # Reduced iterations for larger sizes
    
#     results = []
    
#     print("Starting scaling tests...")
    
#     for size in sizes:
#         size_results = []
#         base_time = None
        
#         for processes in process_counts:
#             print(f"\nTesting size {size}x{size} with {processes} processes")
#             runtime = run_simulation(processes, size, iterations)
            
#             if runtime is not None:
#                 if base_time is None:
#                     base_time = runtime
#                 speedup = base_time / runtime
#                 efficiency = speedup / processes
                
#                 result = {
#                     'size': size,
#                     'processes': processes,
#                     'time': runtime,
#                     'speedup': speedup,
#                     'efficiency': efficiency
#                 }
#                 size_results.append(result)
#                 print(f"Time: {runtime:.3f}s, Speedup: {speedup:.2f}, Efficiency: {efficiency:.2f}")
#             else:
#                 print(f"Failed to get valid result for size={size}, processes={processes}")
        
#         results.extend(size_results)
    
#     return results

# def plot_results(results):
#     """Create plots from test results."""
#     # Group results by size
#     sizes = sorted(set(r['size'] for r in results))
    
#     # Speedup plot
#     plt.figure(figsize=(10, 5))
#     for size in sizes:
#         size_results = [r for r in results if r['size'] == size]
#         processes = [r['processes'] for r in size_results]
#         speedups = [r['speedup'] for r in size_results]
#         plt.plot(processes, speedups, 'o-', label=f'Size {size}x{size}')
    
#     plt.plot([1, max(processes)], [1, max(processes)], 'k--', label='Ideal')
#     plt.xlabel('Number of Processes')
#     plt.ylabel('Speedup')
#     plt.title('Strong Scaling: Speedup vs Processes')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('speedup.png')
#     plt.close()
    
#     # Efficiency plot
#     plt.figure(figsize=(10, 5))
#     for size in sizes:
#         size_results = [r for r in results if r['size'] == size]
#         processes = [r['processes'] for r in size_results]
#         efficiencies = [r['efficiency'] for r in size_results]
#         plt.plot(processes, efficiencies, 'o-', label=f'Size {size}x{size}')
    
#     plt.xlabel('Number of Processes')
#     plt.ylabel('Parallel Efficiency')
#     plt.title('Strong Scaling: Efficiency vs Processes')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('efficiency.png')
#     plt.close()

# def save_results(results):
#     """Save results to file."""
#     with open('scaling_results.txt', 'w') as f:
#         f.write("Lebwohl-Lasher MPI Scaling Results\n")
#         f.write("================================\n\n")
        
#         # Group by size
#         sizes = sorted(set(r['size'] for r in results))
#         for size in sizes:
#             f.write(f"\nResults for {size}x{size} lattice:\n")
#             f.write("-" * 50 + "\n")
#             f.write(f"{'Processes':>10} {'Time (s)':>10} {'Speedup':>10} {'Efficiency':>10}\n")
            
#             size_results = [r for r in results if r['size'] == size]
#             for r in size_results:
#                 f.write(f"{r['processes']:10d} {r['time']:10.3f} {r['speedup']:10.2f} "
#                        f"{r['efficiency']:10.2f}\n")

# def main():
#     """Run full performance evaluation."""
#     print("Starting Lebwohl-Lasher MPI Performance Evaluation")
    
#     # Run scaling tests
#     results = run_scaling_tests()
    
#     if results:
#         # Create plots and save results
#         plot_results(results)
#         save_results(results)
#         print("\nEvaluation completed successfully!")
#         print("Results saved to 'scaling_results.txt'")
#         print("Plots saved as 'speedup.png' and 'efficiency.png'")
#     else:
#         print("\nNo valid results obtained from testing")

# if __name__ == "__main__":
#     main()


"""
Performance evaluation script for MPI Lebwohl-Lasher testing up to 16 processes.
Processes are separate MPI ranks, each running independently.
"""

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path

def run_simulation(processes, size, iterations=50, temp=0.5, timeout=600):
    """Run a single simulation with given parameters."""
    cmd = [
        'mpiexec',
        '--oversubscribe',  # Allow more processes than physical cores
        '-n', str(processes),
        sys.executable,
        'LebwohlLasher_mpi.py',
        str(iterations),
        str(size),
        str(temp),
        '0'
    ]
    
    print(f"\nRunning simulation:")
    print(f"Number of MPI processes: {processes}")
    print(f"Lattice size: {size}x{size}")
    print(f"MC iterations: {iterations}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, 
                              capture_output=True,
                              text=True,
                              timeout=timeout)
        
        if result.returncode != 0:
            print("Error in simulation:")
            print(result.stderr)
            return None
            
        # Extract runtime
        for line in result.stdout.split('\n'):
            if "Time:" in line:
                runtime = float(line.split("Time:")[1].split()[0])
                print(f"Completed in {runtime:.3f} seconds")
                return runtime
                
    except subprocess.TimeoutExpired:
        print(f"Simulation timed out after {timeout} seconds")
    except Exception as e:
        print(f"Error: {e}")
    
    return None

def run_scaling_tests():
    """Run comprehensive scaling tests."""
    # Test parameters
    sizes = [100, 200, 400]  # Larger sizes for better scaling
    process_counts = [1, 2, 4, 8, 16]  # Up to 16 processes
    iterations = 50  # Moderate number of iterations
    
    results = []
    
    print("Starting MPI scaling tests...")
    print(f"Testing lattice sizes: {sizes}")
    print(f"Number of processes: {process_counts}")
    print(f"MPI tasks will be distributed across available cores")
    
    for size in sizes:
        size_results = []
        base_time = None
        
        print(f"\n{'-'*60}")
        print(f"Testing lattice size: {size}x{size}")
        print(f"{'-'*60}")
        
        for processes in process_counts:
            print(f"\nTesting with {processes} MPI processes...")
            runtime = run_simulation(processes, size, iterations)
            
            if runtime is not None:
                if base_time is None:
                    base_time = runtime
                speedup = base_time / runtime
                efficiency = speedup / processes
                
                result = {
                    'size': size,
                    'processes': processes,
                    'time': runtime,
                    'speedup': speedup,
                    'efficiency': efficiency
                }
                size_results.append(result)
                print(f"Runtime: {runtime:.3f} seconds")
                print(f"Speedup: {speedup:.2f}x")
                print(f"Parallel efficiency: {efficiency:.2%}")
            else:
                print(f"Failed to get valid result for size={size}, processes={processes}")
                break
        
        results.extend(size_results)
        
        # Save intermediate results
        save_results(results)
        plot_results(results)
    
    return results

def plot_results(results):
    """Create plots from test results."""
    # Group results by size
    sizes = sorted(set(r['size'] for r in results))
    max_processes = max(r['processes'] for r in results)
    
    # Speedup plot
    plt.figure(figsize=(12, 8))
    for size in sizes:
        size_results = [r for r in results if r['size'] == size]
        processes = [r['processes'] for r in size_results]
        speedups = [r['speedup'] for r in size_results]
        plt.plot(processes, speedups, 'o-', label=f'Size {size}x{size}')
    
    # Ideal scaling line
    x = np.linspace(1, max_processes, 100)
    plt.plot(x, x, 'k--', label='Ideal scaling')
    
    plt.xlabel('Number of MPI Processes')
    plt.ylabel('Speedup')
    plt.title('Strong Scaling: Speedup vs Number of Processes')
    plt.legend()
    plt.grid(True)
    plt.savefig('speedup.png')
    plt.close()
    
    # Efficiency plot
    plt.figure(figsize=(12, 8))
    for size in sizes:
        size_results = [r for r in results if r['size'] == size]
        processes = [r['processes'] for r in size_results]
        efficiencies = [r['efficiency'] for r in size_results]
        plt.plot(processes, efficiencies, 'o-', label=f'Size {size}x{size}')
    
    plt.xlabel('Number of MPI Processes')
    plt.ylabel('Parallel Efficiency')
    plt.title('Strong Scaling: Efficiency vs Number of Processes')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Perfect efficiency')
    
    plt.savefig('efficiency.png')
    plt.close()

def save_results(results):
    """Save results to file."""
    with open('scaling_results.txt', 'w') as f:
        f.write("Lebwohl-Lasher MPI Scaling Results\n")
        f.write("================================\n\n")
        
        f.write("Testing Configuration:\n")
        f.write("- Using MPI for parallel processing\n")
        f.write("- Each process runs on a separate CPU core when available\n")
        f.write("- Processes communicate via MPI messages\n\n")
        
        # Group by size
        sizes = sorted(set(r['size'] for r in results))
        for size in sizes:
            f.write(f"\nResults for {size}x{size} lattice:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Processes':>10} {'Time (s)':>12} {'Speedup':>10} {'Efficiency':>12}\n")
            
            size_results = [r for r in results if r['size'] == size]
            for r in size_results:
                f.write(f"{r['processes']:10d} {r['time']:12.3f} {r['speedup']:10.2f} "
                       f"{r['efficiency']:11.2%}\n")

def main():
    """Run full performance evaluation."""
    print("Starting Lebwohl-Lasher MPI Performance Evaluation")
    print("Testing parallel scaling up to 16 MPI processes")
    
    # Run scaling tests
    results = run_scaling_tests()
    
    if results:
        print("\nEvaluation completed successfully!")
        print("Results saved to 'scaling_results.txt'")
        print("Plots saved as 'speedup.png' and 'efficiency.png'")
        
        # Print summary of best results
        print("\nBest speedups achieved:")
        sizes = sorted(set(r['size'] for r in results))
        for size in sizes:
            size_results = [r for r in results if r['size'] == size]
            best_speedup = max(r['speedup'] for r in size_results)
            best_process = max((r for r in size_results), key=lambda x: x['speedup'])
            print(f"Size {size}x{size}: {best_speedup:.2f}x speedup with {best_process['processes']} processes "
                  f"(efficiency: {best_process['efficiency']:.2%})")
    else:
        print("\nNo valid results obtained from testing")

if __name__ == "__main__":
    main()
