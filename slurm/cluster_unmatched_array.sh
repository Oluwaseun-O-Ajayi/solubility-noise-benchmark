#!/bin/bash
#SBATCH --job-name=solnoise_unmatched
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-99%20
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=oluwaseun.ajayi@uga.edu

# One array task = one row of grids/cluster_unmatched.csv (100 rows: 80 ESOL, 20 AqSolDB).
# The cluster split without label matching, used as a sensitivity control.
cd $SLURM_SUBMIT_DIR
source $HOME/envs/solnoise/bin/activate
export OMP_NUM_THREADS=4
python scripts/run_benchmark.py --grid grids/cluster_unmatched.csv --task $SLURM_ARRAY_TASK_ID \
    --out results/v2 --threads 1 --skip-existing
