"""
Script to plot benchmark results from HPC runs
Combines results from different process counts
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_all_results():
    """Load benchmark results for all process counts"""
    results_dir = Path('benchmark_results')
    all_results = []
    
    # Load results for each process count
    for p in [1, 2, 4, 8, 16]:
        result_file = results_dir / f'results_p{p:02d}.json'
        if result_file.exists():
            with open(result_file, 'r') as f:
                all_results.extend(json.load(f))
    
    return all_results

def create_plots(results):
    """Generate comprehensive plots from benchmark results"""
    # Group results by process count and size
    process_counts = sorted(set(r['processes'] for r in results))
    sizes = sorted(set(r['size'] for r in results))
    
    plt.style.use('default')
    
    # Plot 1: Strong Scaling - Time vs Processes for each size
    plt.figure(figsize=(12, 8))
    for size in sizes:
        size_results = [r for r in results if r['size'] == size]
        if len(size_results) > 1:  # Only plot if we have multiple process counts
            procs = [r['processes'] for r in size_results]
            times = [r['time_per_step'] for r in size_results]
            plt.loglog(procs, times, 'o-', label=f'{size}x{size}')
    
    # Add ideal scaling line
    plt.loglog(process_counts, [times[0]/p for p in process_counts], 'k--', 
               label='Ideal Scaling')
    
    plt.grid(True, which="both", ls="-")
    plt.xlabel('Number of Processes')
    plt.ylabel('Time per Step (seconds)')
    plt.title('Strong Scaling: Performance vs Process Count')
    plt.legend(title='Lattice Size', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('benchmark_results/strong_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Parallel Efficiency
    plt.figure(figsize=(12, 8))
    for size in sizes:
        size_results = [r for r in results if r['size'] == size]
        if len(size_results) > 1:
            procs = [r['processes'] for r in size_results]
            times = [r['time_per_step'] for r in size_results]
            # Calculate efficiency relative to single process
            base_time = times[0]
            efficiency = [base_time/(p*t) for p, t in zip(procs, times)]
            plt.semilogx(procs, efficiency, 'o-', label=f'{size}x{size}')
    
    plt.grid(True)
    plt.xlabel('Number of Processes')
    plt.ylabel('Parallel Efficiency')
    plt.title('Strong Scaling: Efficiency vs Process Count')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Ideal Efficiency')
    plt.legend(title='Lattice Size', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('benchmark_results/parallel_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Problem Size Scaling for each process count
    plt.figure(figsize=(12, 8))
    for nprocs in process_counts:
        proc_results = [r for r in results if r['processes'] == nprocs]
        if proc_results:
            sizes = [r['size'] for r in proc_results]
            times = [r['time_per_step'] for r in proc_results]
            plt.loglog(sizes, times, 'o-', label=f'{nprocs} processes')
    
    plt.grid(True, which="both", ls="-")
    plt.xlabel('Lattice Size (NxN)')
    plt.ylabel('Time per Step (seconds)')
    plt.title('Weak Scaling: Performance vs Problem Size')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('benchmark_results/size_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results
    with open('benchmark_results/scaling_analysis.txt', 'w') as f:
        f.write("Lebwohl-Lasher Parallel Scaling Analysis\n")
        f.write("=====================================\n\n")
        
        # Strong scaling analysis
        f.write("Strong Scaling Analysis:\n")
        f.write("----------------------\n")
        for size in sorted(set(r['size'] for r in results)):
            size_results = [r for r in results if r['size'] == size]
            if len(size_results) > 1:
                f.write(f"\nLattice Size {size}x{size}:\n")
                base_time = size_results[0]['time_per_step']
                for r in size_results:
                    speedup = base_time / r['time_per_step']
                    efficiency = speedup / r['processes']
                    f.write(f"  {r['processes']} processes: {r['time_per_step']:.3f} s ")
                    f.write(f"(speedup: {speedup:.2f}x, efficiency: {efficiency:.2%})\n")

def main():
    """Main function to create all plots and analysis"""
    results = load_all_results()
    if not results:
        print("No results files found!")
        return
    
    create_plots(results)
    print("Analysis complete. Plots and analysis saved in benchmark_results/")

if __name__ == "__main__":
    main()