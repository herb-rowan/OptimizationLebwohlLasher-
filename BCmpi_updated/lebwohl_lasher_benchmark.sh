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

# Load required modules
module purge
module load languages/python/3.12.3

cd $SLURM_SUBMIT_DIR

# Create results directory if it doesn't exist
mkdir -p benchmark_results

# Test more process counts but with optimized sizes
for NPROCS in 1 2 4 8 16
do
    echo "Running benchmark with $NPROCS processes"
    mpiexec -n $NPROCS python ll_benchmark_hpc.py
    sleep 2  # Short pause between runs
done

# Generate plots from results
python plot_benchmark_results.py