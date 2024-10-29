"""
Script to plot benchmark results from HPC runs
Focusing on lattice size scaling for different process counts
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def load_all_results():
    """Load benchmark results for all process counts with error handling"""
    results_dir = Path('benchmark_results')
    all_results = []
    
    # Check if directory exists
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} not found!")
        return []
    
    # Load results for each process count
    for p in [1, 2, 4, 8, 16]:
        result_file = results_dir / f'results_p{p:02d}.json'
        print(f"\nTrying to load: {result_file}")
        
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    content = f.read()
                    if content.strip():
                        data = json.loads(content)
                        all_results.extend(data)
                    else:
                        print(f"Warning: Empty file for {p} processes")
            except json.JSONDecodeError as e:
                print(f"Error reading results for {p} processes: {e}")
            except Exception as e:
                print(f"Unexpected error reading file for {p} processes: {e}")
        else:
            print(f"Warning: No results file found for {p} processes")
    
    return all_results

def create_plots(results):
    """Generate plot of time vs lattice size for different process counts"""
    if not results:
        print("No results to plot!")
        return
    
    # Group results by process count
    process_counts = sorted(set(r['processes'] for r in results))
    
    # Create the main plot
    plt.figure(figsize=(12, 8))
    
    # Set up grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Different marker styles for each process count
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot data for each process count
    for i, nprocs in enumerate(process_counts):
        proc_results = [r for r in results if r['processes'] == nprocs]
        if proc_results:
            sizes = [r['size'] for r in proc_results]
            times = [r['total_time'] for r in proc_results]
            
            plt.loglog(sizes, times, 
                      marker=markers[i % len(markers)],
                      color=colors[i % len(colors)],
                      linestyle='-',
                      linewidth=2,
                      markersize=8,
                      label=f'{nprocs} process{"es" if nprocs > 1 else ""}')
    
    # Add N² scaling reference line
    if results:
        min_size = min(r['size'] for r in results)
        max_size = max(r['size'] for r in results)
        ref_sizes = np.logspace(np.log10(min_size), np.log10(max_size), 100)
        # Scale to match at the smallest size
        single_proc_results = [r for r in results if r['processes'] == 1]
        if single_proc_results:
            ref_point = min((r for r in single_proc_results if r['size'] == min_size), 
                          key=lambda x: x['total_time'])
            ref_time = ref_point['total_time']
            ref_scale = ref_time / (min_size * min_size)
            ref_times = ref_scale * ref_sizes * ref_sizes
            plt.loglog(ref_sizes, ref_times, 'k--', label='N² scaling', alpha=0.5)
    
    plt.xlabel('Lattice Size (N×N)', fontsize=12)
    plt.ylabel('Total Time (seconds)', fontsize=12)
    plt.title('Lebwohl-Lasher Performance Scaling', fontsize=14, pad=20)
    
    # Adjust legend
    plt.legend(title='Number of Processes', 
              title_fontsize=12,
              fontsize=10,
              bbox_to_anchor=(1.05, 1),
              loc='upper left')
    
    # Ensure all markers are visible
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('benchmark_results/size_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results
    with open('benchmark_results/size_scaling_analysis.txt', 'w') as f:
        f.write("Lebwohl-Lasher Size Scaling Analysis\n")
        f.write("=================================\n\n")
        
        # Sort by size first, then by process count
        sizes = sorted(set(r['size'] for r in results))
        
        for size in sizes:
            f.write(f"\nLattice Size {size}x{size}:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Processes':>10} {'Time (s)':>12} {'Iterations':>12}\n")
            
            size_results = sorted([r for r in results if r['size'] == size],
                                key=lambda x: x['processes'])
            
            for r in size_results:
                f.write(f"{r['processes']:10d} {r['total_time']:12.3f} {r['iterations']:12d}\n")
            
            if size_results:
                # Calculate speedup relative to single process
                single_proc = next((r for r in size_results if r['processes'] == 1), None)
                if single_proc:
                    f.write("\nSpeedup Analysis:\n")
                    base_time = single_proc['total_time']
                    for r in size_results:
                        speedup = base_time / r['total_time']
                        efficiency = speedup / r['processes']
                        f.write(f"{r['processes']:3d} processes: {speedup:6.2f}x speedup "
                               f"({efficiency:6.1%} efficiency)\n")
            f.write("\n")

def main():
    """Main function to create all plots and analysis"""
    print("Starting benchmark analysis...")
    
    results = load_all_results()
    if not results:
        print("Error: No valid results found to analyze!")
        sys.exit(1)
    
    create_plots(results)
    print("\nAnalysis complete. Check benchmark_results/ directory for outputs.")

if __name__ == "__main__":
    main()