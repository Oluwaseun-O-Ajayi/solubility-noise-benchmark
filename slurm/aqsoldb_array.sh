#!/bin/bash
#SBATCH --job-name=solnoise_aqsoldb
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-59%20
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=oluwaseun.ajayi@uga.edu

# One array task = one row of grids/aqsoldb.csv (60 rows: 3 splits x 4 noise settings x 5 seeds).
# The Gaussian processes use 4 BLAS threads; the networks stay on 1 PyTorch thread for reproducibility.
cd $SLURM_SUBMIT_DIR
source $HOME/envs/solnoise/bin/activate
export OMP_NUM_THREADS=4
python scripts/run_benchmark.py --grid grids/aqsoldb.csv --task $SLURM_ARRAY_TASK_ID \
    --out results/v2 --threads 1 --skip-existing
