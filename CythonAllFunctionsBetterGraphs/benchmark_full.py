# import numpy as np
# import time
# import matplotlib.pyplot as plt
# from LebwohlLasher import one_energy as one_energy_py, MC_step as MC_step_py, get_order as get_order_py
# from LebwohlLasher_full import one_energy as one_energy_cy, MC_step as MC_step_cy, get_order as get_order_cy

# def benchmark_all_functions(nmax, n_tests=100):
#     # Create test array
#     arr = np.random.random((nmax, nmax)) * 2.0 * np.pi
#     Ts = 0.5
    
#     # Test points for one_energy
#     test_points = [(np.random.randint(1, nmax-1), np.random.randint(1, nmax-1)) 
#                    for _ in range(n_tests)]
    
#     # Benchmark one_energy
#     start = time.time()
#     for ix, iy in test_points:
#         one_energy_py(arr, ix, iy, nmax)
#     py_energy_time = time.time() - start
    
#     start = time.time()
#     for ix, iy in test_points:
#         one_energy_cy(arr, ix, iy, nmax)
#     cy_energy_time = time.time() - start
    
#     # Benchmark MC_step
#     start = time.time()
#     for _ in range(n_tests):
#         MC_step_py(arr, Ts, nmax)
#     py_mc_time = time.time() - start
    
#     start = time.time()
#     for _ in range(n_tests):
#         MC_step_cy(arr, Ts, nmax)
#     cy_mc_time = time.time() - start
    
#     # Benchmark get_order
#     start = time.time()
#     for _ in range(n_tests):
#         get_order_py(arr, nmax)
#     py_order_time = time.time() - start
    
#     start = time.time()
#     for _ in range(n_tests):
#         get_order_cy(arr, nmax)
#     cy_order_time = time.time() - start
    
#     return {
#         'energy': (py_energy_time, cy_energy_time),
#         'mc': (py_mc_time, cy_mc_time),
#         'order': (py_order_time, cy_order_time)
#     }

# def plot_results(sizes, results):
#     functions = ['Energy Calculation', 'Monte Carlo Step', 'Order Parameter']
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
#     for idx, func in enumerate(['energy', 'mc', 'order']):
#         py_times = [results[size][func][0] for size in sizes]
#         cy_times = [results[size][func][1] for size in sizes]
#         speedup = [py/cy for py, cy in zip(py_times, cy_times)]
        
#         axes[idx].plot(sizes, py_times, 'o-', label='Python')
#         axes[idx].plot(sizes, cy_times, 'o-', label='Cython')
#         axes[idx].set_xlabel('Lattice Size')
#         axes[idx].set_ylabel('Time (s)')
#         axes[idx].set_title(f'{functions[idx]}\nSpeedup: {np.mean(speedup):.1f}x')
#         axes[idx].legend()
#         axes[idx].set_xscale('log')
#         axes[idx].set_yscale('log')
#         axes[idx].grid(True)
    
#     plt.tight_layout()
#     plt.savefig('cython_performance.png')
#     plt.show()

# if __name__ == "__main__":
#     sizes = [10, 20, 50, 100, 200]
#     all_results = {}
    
#     for size in sizes:
#         print(f"Testing lattice size {size}...")
#         all_results[size] = benchmark_all_functions(size)
    
#     plot_results(sizes, all_results)

import numpy as np
import time
import matplotlib.pyplot as plt
from tabulate import tabulate  # You might need to install this: pip install tabulate
from LebwohlLasher import one_energy as one_energy_py, MC_step as MC_step_py, get_order as get_order_py
from LebwohlLasher_full import one_energy as one_energy_cy, MC_step as MC_step_cy, get_order as get_order_cy

def benchmark_all_functions(nmax, n_tests=100):
    # Create test array
    arr = np.random.random((nmax, nmax)) * 2.0 * np.pi
    arr = arr.astype(np.float64)  # Ensure double precision
    Ts = 0.5
    
    results = {}
    
    # Benchmark one_energy
    print("Testing one_energy...")
    test_points = [(np.random.randint(1, nmax-1), np.random.randint(1, nmax-1)) 
                   for _ in range(n_tests)]
    
    start = time.time()
    for ix, iy in test_points:
        one_energy_py(arr, ix, iy, nmax)
    py_energy_time = time.time() - start
    
    start = time.time()
    for ix, iy in test_points:
        one_energy_cy(arr, ix, iy, nmax)
    cy_energy_time = time.time() - start
    
    results['energy'] = (py_energy_time, cy_energy_time)
    
    # Benchmark MC_step
    print("Testing MC_step...")
    start = time.time()
    for _ in range(n_tests):
        MC_step_py(arr.copy(), Ts, nmax)  # Use copy to prevent modification
    py_mc_time = time.time() - start
    
    start = time.time()
    for _ in range(n_tests):
        MC_step_cy(arr.copy(), Ts, nmax)  # Use copy to prevent modification
    cy_mc_time = time.time() - start
    
    results['mc'] = (py_mc_time, cy_mc_time)
    
    # Benchmark get_order
    print("Testing get_order...")
    start = time.time()
    for _ in range(n_tests):
        get_order_py(arr, nmax)
    py_order_time = time.time() - start
    
    start = time.time()
    for _ in range(n_tests):
        get_order_cy(arr, nmax)
    cy_order_time = time.time() - start
    
    results['order'] = (py_order_time, cy_order_time)
    
    return results

def create_results_table(sizes, results):
    table_data = []
    headers = ["Lattice Size", "Function", "Python (s)", "Cython (s)", "Speedup"]
    
    for size in sizes:
        for func in ['energy', 'mc', 'order']:
            py_time, cy_time = results[size][func]
            speedup = py_time / cy_time
            func_name = {
                'energy': 'Energy Calculation',
                'mc': 'Monte Carlo Step',
                'order': 'Order Parameter'
            }[func]
            table_data.append([
                f"{size}×{size}",
                func_name,
                f"{py_time:.6f}",
                f"{cy_time:.6f}",
                f"{speedup:.1f}×"
            ])
    
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    
    # Save table to file
    with open('performance_results.txt', 'w') as f:
        f.write(table)
    
    return table

def plot_results(sizes, results):
    functions = ['Energy Calculation', 'Monte Carlo Step', 'Order Parameter']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, func in enumerate(['energy', 'mc', 'order']):
        py_times = [results[size][func][0] for size in sizes]
        cy_times = [results[size][func][1] for size in sizes]
        speedup = [py/cy for py, cy in zip(py_times, cy_times)]
        
        axes[idx].plot(sizes, py_times, 'o-', label='Python')
        axes[idx].plot(sizes, cy_times, 'o-', label='Cython')
        axes[idx].set_xlabel('Lattice Size')
        axes[idx].set_ylabel('Time (s)')
        axes[idx].set_title(f'{functions[idx]}\nSpeedup: {np.mean(speedup):.1f}x')
        axes[idx].legend()
        axes[idx].set_xscale('log')
        axes[idx].set_yscale('log')
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.savefig('cython_performance.png')
    plt.show()

if __name__ == "__main__":
    sizes = [10, 20, 50, 100, 200]
    all_results = {}
    
    for size in sizes:
        print(f"\nTesting lattice size {size}...")
        try:
            all_results[size] = benchmark_all_functions(size)
            print(f"Completed size {size}")
        except Exception as e:
            print(f"Error testing size {size}: {str(e)}")
    
    if all_results:
        # Create and display table
        print("\nDetailed Performance Results:")
        print(create_results_table(sizes, all_results))
        
        # Create plot
        plot_results(sizes, all_results)
    else:
        print("No results to plot!")