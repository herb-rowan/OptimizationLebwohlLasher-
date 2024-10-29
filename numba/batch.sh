#!/bin/bash
# ======================
# pythonserialscript.sh
# ======================

#SBATCH --job-name=test_job
#SBATCH --partition=teach_cpu
#SBATCH --account=PHYS033185
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:0:10
#SBATCH --mem=100M

# Load modules required for runtime e.g.
module add languages/python/3.12.3

cd $SLURM_SUBMIT_DIR

# Now run your program with the usual command
python picalc_py.py 10000000
