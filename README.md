# OptimizationLebwohlLasher

This guide provides step-by-step instructions to set up and run the files for the Lebwohl-Lasher model project.

# CythonOneEnergy:

# Instructions to Run the CythonOneEnergy Folder

## Project Files Overview

1. **`LebwohlLasher.py`** - Contains the main code for the Lebwohl-Lasher model.
2. **`LebwohlLasher_cython_test.pyx`** - A Cython implementation of the Lebwohl-Lasher model to improve performance.
3. **`benchmark.py`** - A benchmarking script to compare the performance of different implementations.
4. **`setup.py`** - A setup file to compile the Cython code.

## Requirements

Before you begin, make sure you have the following:

1. **Python 3.10** - As you're using a MacBook Air with Python 3.10.
2. **Cython** - Required to compile the `.pyx` file.
3. **NumPy** - Required for numerical computations.
4. **Compiler** - A C/C++ compiler (e.g., `gcc` or `clang`) to compile the Cython extension.

### Install Required Packages

Use the following command to install the necessary packages:


pip install numpy cython
## Steps to Run the Project

### 1. Build the Cython Extension
Since `LebwohlLasher_cython_test.pyx` is a Cython file, it needs to be compiled before use.

- Open a terminal in the directory where the files are located.
- Run the following command to compile the Cython extension:

    ```bash
    python setup.py build_ext --inplace
    ```

   This command will generate a compiled `.so` file (a shared object file) for `LebwohlLasher_cython_test`, allowing it to be imported as a Python module.

### 2. Run the Benchmark Script
Once the Cython module is built, you can run the `benchmark.py` script to test the performance of the model.

- In the terminal, execute:

    ```bash
    python benchmark.py
    ```

   This script will likely output performance metrics, allowing you to compare the Cython-optimized model to the standard Python implementation.

### 3. Run the Main Model Script (Optional)
If `LebwohlLasher.py` includes a main function or similar executable code, you can run it directly to test the model's functionality:

    ```bash
    python LebwohlLasher.py
    ```

   This will allow you to see the model’s behavior independently of the benchmarking script if needed.

# Numba

# Instructions to Run the Lebwohl-Lasher Model Project on an HPC Cluster

This guide provides instructions for running the Lebwohl-Lasher model files on an HPC cluster, using the provided SLURM batch script.

## Files Overview

1. **`LebwohlLasher.py`** - Contains the main code for the Lebwohl-Lasher model.
2. **`LebwohlLasherNumba.py`** - An optimized Numba implementation of the Lebwohl-Lasher model.
3. **`benchmark_ll.py`** - A script to benchmark the performance of different implementations.
4. **`batch.sh`** - A SLURM batch script for submitting the job on an HPC cluster.

## Requirements

Ensure the HPC environment has:
- **Python 3.12.3** (or compatible version)
- **NumPy** and **Numba** packages

## Steps to Run the Project

### 1. Update `batch.sh`

Make sure `batch.sh` is correctly set up for your file and parameters.



### 2. Submit the Job to the Cluster

- Open a terminal and navigate to the directory containing `batch.sh`.
- Submit the job with the following command:


    sbatch batch.sh


This will send the job to the scheduler, which will queue and execute it on the cluster.

### 3. Check Job Status

To monitor the status of your job, use:


squeue -u your_username


# mpi_numpy folder

## Files Overview

1. **`LebwohlLasher_mpi_sequential.py`** - A sequential implementation of the Lebwohl-Lasher model using MPI.
2. **`ll_benchmark_hpc_vectorized.py`** - A vectorized MPI benchmark script.
3. **`plot_benchmark_results.py`** - A script to visualize benchmark results.
4. **`lebwohl_lasher_benchmark.sh`** - A SLURM batch script for submitting the benchmark job on an HPC cluster.

## Requirements

Ensure the HPC environment has:
- **Python 3.12.3** (or compatible version)
- **NumPy** for numerical operations
- **MPI** support for parallel processing (e.g., `mpiexec` command)

## Steps to Run the Project

### 1. Review and Customize `lebwohl_lasher_benchmark.sh`

Ensure that `lebwohl_lasher_benchmark.sh` is configured correctly for your needs, including job name, partition, memory, and time limits. This script will execute the benchmark with different numbers of MPI processes and generate benchmark results.

- **Example `lebwohl_lasher_benchmark.sh` setup**:


   

### 2. Submit the Job to the Cluster

Navigate to the directory containing `lebwohl_lasher_benchmark.sh` and submit the job using:


sbatch lebwohl_lasher_benchmark.sh

# BCmpi_updated

# Instructions to Run the Lebwohl-Lasher Benchmark Project on an HPC Cluster

## Files Overview

1. **`LebwohlLasher_mpi.py`** - An MPI implementation of the Lebwohl-Lasher model.
2. **`ll_benchmark_hpc.py`** - An MPI benchmark script for testing performance.
3. **`plot_benchmark_results.py`** - A script to visualize benchmark results.
4. **`lebwohl_lasher_benchmark_long.sh`** - A SLURM batch script for submitting the benchmark job on an HPC cluster.

## Requirements

Ensure the HPC environment has:
- **Python 3.12.3** (or compatible version)
- **NumPy** for numerical operations
- **MPI** support for parallel processing (e.g., `mpiexec` command)

## Steps to Run the Project

### 1. Review and Customize `lebwohl_lasher_benchmark_long.sh`

Ensure that `lebwohl_lasher_benchmark_long.sh` is configured correctly for your needs, including job name, partition, memory, and time limits. This script will execute the benchmark with different numbers of MPI processes and generate benchmark results.

- **Example `lebwohl_lasher_benchmark_long.sh` setup**:

   
### 2. Submit the Job to the Cluster

Navigate to the directory containing `lebwohl_lasher_benchmark_long.sh` and submit the job using:


sbatch lebwohl_lasher_benchmark_long.sh


# Cython All Functions

# Instructions to Run the Lebwohl-Lasher Full Benchmark Project

This guide provides instructions for running the Lebwohl-Lasher full benchmark files on a local machine.

## Files Overview

1. **`LebwohlLasher_full.pyx`** - A Cython implementation of the Lebwohl-Lasher model to improve performance.
2. **`LebwohlLasher.py`** - Contains the main code for the Lebwohl-Lasher model.
3. **`benchmark_full.py`** - A benchmarking script to test the performance of the full model implementation.
4. **`setup.py`** - A setup file to compile the Cython code.

## Requirements

Ensure your environment has:
- **Python 3.12.3** (or compatible version)
- **NumPy** for numerical operations
- **Cython** to compile `.pyx` files
- **Compiler** (e.g., `gcc` or `clang`) for Cython compilation

## Steps to Run the Project

### 1. Compile the Cython Extension

You need to compile `LebwohlLasher_full.pyx` to generate a `.so` file for the Cython extension.

- Open a terminal in the directory containing `setup.py`.
- Run the following command:


    python setup.py build_ext --inplace
    ```

   This will generate a compiled `.so` file, allowing `LebwohlLasher_full` to be used as a module in Python.

### 2. Run the Benchmark Script

After compiling the Cython extension, you can run the benchmark script to test performance.

In the terminal, run:


python benchmark_full.py


# CythonOpenMPOneRun

# Instructions to Run the Lebwohl-Lasher Parallel Timing Project

This guide provides instructions for running the Lebwohl-Lasher parallel timing files on a local machine.

## Files Overview

1. **`ll_parallel.pyx`** - A Cython file containing the parallel implementation of the Lebwohl-Lasher model for performance optimization.
2. **`run_parallel_timing.py`** - A script to measure the timing and performance of the parallel model implementation.
3. **`setup.py`** - A setup file to compile the Cython code.

## Requirements

Ensure your environment has:
- **Python 3.12.3** (or compatible version)
- **NumPy** for numerical operations
- **Cython** to compile `.pyx` files
- **Compiler** (e.g., `gcc` or `clang`) for Cython compilation

## Steps to Run the Project

### 1. Compile the Cython Extension

You need to compile `ll_parallel.pyx` to generate a `.so` file for the Cython extension.

- Open a terminal in the directory containing `setup.py`.
- Run the following command:


    python setup.py build_ext --inplace
    ```

   This will generate a compiled `.so` file, allowing `ll_parallel` to be used as a module in Python.

### 2. Run the Timing Script

Once the Cython extension is compiled, you can run the timing script to measure the performance of the parallel implementation.

In the terminal, execute:


python run_parallel_timing.py


### There are also some testing scripts, test_mpi, lebwohlasher_test and a .github continugous testing folder, all can be adapted to specific needs. 

### InitialAnalysis replicates the results from the report and performs some profiling. 