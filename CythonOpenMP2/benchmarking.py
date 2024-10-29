# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import subprocess
# import os
# import sys

# # Add the current directory to sys.path to import monte_carlo
# sys.path.append(os.getcwd())

# # Import the compiled Cython module
# try:
#     import monte_carlo
# except ImportError as e:
#     print("Failed to import monte_carlo module. Make sure it is compiled.")
#     raise e

# # Define the range of lattice sizes to test
# lattice_sizes = [50, 100, 150, 200, 250, 300]

# # Define the number of threads to test with OpenMP
# thread_counts = [1, 2, 4, 8, 16]

# # Number of Monte Carlo steps and temperature (example values)
# nsteps = 100
# temperature = 1.0

# # Initialize dictionaries to store execution times
# python_times = []
# cython_times = {threads: [] for threads in thread_counts}

# # Benchmark the original Python code
# for nmax in lattice_sizes:
#     start_time = time.time()
#     # Provide all required arguments, including PLOTFLAG (set to 0)
#     subprocess.run(['python3.10', 'LebwohlLasher.py', str(nsteps), str(nmax), str(temperature), '0'], check=True)
#     end_time = time.time()
#     python_times.append(end_time - start_time)
#     print(f'Python: Lattice Size {nmax}, Time {end_time - start_time:.2f} seconds')

# # Benchmark the Cython code with different thread counts
# for threads in thread_counts:
#     for nmax in lattice_sizes:
#         start_time = time.time()
#         # Call the main function of the Cython module
#         # Passing 'threads' as the pflag to set the number of OpenMP threads
#         monte_carlo.main('program', nsteps, nmax, temperature, threads)
#         end_time = time.time()
#         cython_times[threads].append(end_time - start_time)
#         print(f'Cython ({threads} threads): Lattice Size {nmax}, Time {end_time - start_time:.2f} seconds')

# # Plotting the results
# plt.figure(figsize=(12, 8))
# plt.plot(lattice_sizes, python_times, label='Python', marker='o')
# for threads in thread_counts:
#     plt.plot(lattice_sizes, cython_times[threads], label=f'Cython ({threads} threads)', marker='o')
# plt.xlabel('Lattice Size')
# plt.ylabel('Execution Time (seconds)')
# plt.title('Performance Comparison: Python vs Cython with OpenMP')
# plt.legend()
# plt.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add the current directory to sys.path to import monte_carlo
sys.path.append(os.getcwd())

# Import the compiled Cython module
try:
    import monte_carlo
except ImportError as e:
    print("Failed to import monte_carlo module. Make sure it is compiled.")
    raise e

# Define the range of lattice sizes to test
lattice_sizes = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000]

# Define the number of threads to test with OpenMP
thread_counts = [1, 2, 4, 8, 16]

# Number of Monte Carlo steps and temperature (example values)
nsteps = 100
temperature = 1.0

# Initialize dictionary to store Cython execution times
cython_times = {threads: [] for threads in thread_counts}

# Benchmark the Cython code with different thread counts
for threads in thread_counts:
    for nmax in lattice_sizes:
        start_time = time.time()
        # Call the main function of the Cython module
        # Passing 'threads' as the pflag to set the number of OpenMP threads
        monte_carlo.main('program', nsteps, nmax, temperature, threads)
        end_time = time.time()
        cython_times[threads].append(end_time - start_time)
        print(f'Cython ({threads} threads): Lattice Size {nmax}, Time {end_time - start_time:.2f} seconds')

# Plotting the results
plt.figure(figsize=(12, 8))
for threads in thread_counts:
    plt.plot(lattice_sizes, cython_times[threads], label=f'Cython ({threads} threads)', marker='o')
plt.xlabel('Lattice Size')
plt.ylabel('Execution Time (seconds)')
plt.title('Performance of Cython with OpenMP at Different Thread Counts')
plt.legend()
plt.grid(True)
plt.show()

