#!/bin/bash
#SBATCH --job-name=solnoise_gbt
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-299%40
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=oluwaseun.ajayi@uga.edu

# One array task = one row of grids/gbt.csv (300 rows: every condition of the main grid).
# Only the two gradient-boosting models are fitted; results go to a separate folder
# so the existing results/v2 files are never overwritten. Merge afterwards with
# scripts/merge_results.py.
cd $SLURM_SUBMIT_DIR
source $HOME/envs/solnoise/bin/activate
export OMP_NUM_THREADS=2
python scripts/run_benchmark.py --grid grids/gbt.csv --task $SLURM_ARRAY_TASK_ID \
    --models gbt_desc gbt_fp --out results/v2_gbt --threads 1 --skip-existing
