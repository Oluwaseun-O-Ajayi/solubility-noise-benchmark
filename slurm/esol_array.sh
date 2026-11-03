#!/bin/bash
#SBATCH --job-name=solnoise_esol
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --array=0-239%40
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=oluwaseun.ajayi@uga.edu

# One array task = one row of grids/esol.csv (240 rows: 3 splits x 8 noise settings x 10 seeds).
# %40 runs at most 40 tasks at the same time.
cd $SLURM_SUBMIT_DIR
source $HOME/envs/solnoise/bin/activate
export OMP_NUM_THREADS=1
python scripts/run_benchmark.py --grid grids/esol.csv --task $SLURM_ARRAY_TASK_ID \
    --out results/v2 --threads 1 --skip-existing
