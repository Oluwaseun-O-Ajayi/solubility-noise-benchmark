#!/bin/bash
#SBATCH --job-name=solnoise_test
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# Run this first: one ESOL condition (row 0) to confirm the environment works.
cd $SLURM_SUBMIT_DIR
source $HOME/envs/solnoise/bin/activate
export OMP_NUM_THREADS=1
python -m pytest -q tests/
python scripts/run_benchmark.py --grid grids/esol.csv --task 0 --out results/test_run --threads 1
