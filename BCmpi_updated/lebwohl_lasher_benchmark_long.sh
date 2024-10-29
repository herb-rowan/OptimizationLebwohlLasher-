#!/bin/bash
# =================
# lebwohl_lasher_benchmark.sh
# =================

#SBATCH --job-name=ll_bench
#SBATCH --partition=teach_cpu
#SBATCH --account=PHYS033185
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=2G

# Print debug information
echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -l

# Load required modules
module purge  # Clear any existing modules
module load languages/python/3.12.3
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

cd $SLURM_SUBMIT_DIR
echo "SLURM directory: $SLURM_SUBMIT_DIR"
echo "Directory contents after cd:"
ls -l

# Create results directory if it doesn't exist
mkdir -p benchmark_results

# List all Python scripts to verify they exist
echo "Python scripts in directory:"
ls -l *.py

# Run benchmarks with different numbers of processes
for NPROCS in 1 2 4 8 16
do
    echo "Running benchmark with $NPROCS processes"
    mpiexec -n $NPROCS python ll_benchmark_hpc.py $NPROCS
done

# Generate plots from all results
python plot_benchmark_results.py