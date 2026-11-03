# Running the revised benchmark on Sapelo2

These steps run the full grid on the GACRC Sapelo2 cluster. Commands in boxes are typed on Sapelo2 unless a step says "on your computer".

The full grid is 240 ESOL conditions (3 splits × 8 noise settings × 10 seeds) and 60 AqSolDB conditions (3 splits × 4 noise settings × 5 seeds). An ESOL condition takes about 3 to 10 minutes on one core; an AqSolDB condition takes about 1 to 3 hours on four cores.

## 1. Log in and go to scratch

```bash
ssh ooa97459@sapelo2.gacrc.uga.edu
cd /scratch/ooa97459
```

Jobs run from `/scratch`. Files there are deleted after 30 days without use, so step 7 copies the results out.

## 2. Upload the code

On your computer, in the folder where you saved `solubility-noise-benchmark-revision.zip`:

```bash
scp solubility-noise-benchmark-revision.zip ooa97459@xfer.gacrc.uga.edu:/scratch/ooa97459/
```

Then on Sapelo2:

```bash
cd /scratch/ooa97459
unzip solubility-noise-benchmark-revision.zip
cd solubility-noise-benchmark
```

(Later, once the branch `revision-digital-discovery` is pushed to GitHub, `git clone` and `git checkout revision-digital-discovery` do the same.)

## 3. Make a Python environment (once)

Find a Python module of version 3.10 or later and load it:

```bash
ml spider Python
ml Python/<version from the list>
python -m venv $HOME/envs/solnoise
source $HOME/envs/solnoise/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If installing on the login node is slow or refused, do it inside an interactive session instead (see the GACRC wiki for the current interactive command). The job scripts activate `$HOME/envs/solnoise`, so the environment must be at that path. If you load a Python module in step 3, add the same `ml Python/<version>` line to the three scripts in `slurm/`, just above `source`.

## 4. Test with one job

```bash
mkdir -p logs
sbatch slurm/test_one.sh
squeue --me
```

When it finishes, read the log:

```bash
cat logs/solnoise_test_<jobID>.out
```

You should see `36 passed` from the tests, then one line per model with `rmse=`. If there is an error, stop here and send me the log.

## 5. Submit the full grids

```bash
sbatch slurm/esol_array.sh
sbatch slurm/aqsoldb_array.sh
squeue --me
```

`%40` and `%20` in the scripts limit how many tasks run at once. You get an email when each array ends or a task fails.

## 6. Check progress and rerun anything that failed

```bash
python scripts/collect.py
```

This prints how many conditions of each grid are done and names any that are missing. The scripts use `--skip-existing`, so resubmitting the same array only reruns the missing tasks:

```bash
sbatch slurm/esol_array.sh
```

To see why a task failed, read its log: `logs/solnoise_esol_<arrayID>_<taskID>.out`. If AqSolDB tasks run out of time or memory, raise `--time` or `--mem` in `slurm/aqsoldb_array.sh` and resubmit.

## 7. Copy the results to your computer

On your computer (PowerShell or Terminal):

```bash
scp -r ooa97459@xfer.gacrc.uga.edu:/scratch/ooa97459/solubility-noise-benchmark/results/v2 .
```

Then send me the `v2` folder (zip it first).

## What each file contains

| Path | Content |
|---|---|
| `results/v2/metrics/<condition>.csv` | One row per model: accuracy, coverage (total, epistemic, recalibrated, conformal), NLL, miscalibration area, rank correlation and its simulated reference, error-based calibration, fitted GP noise |
| `results/v2/predictions/<condition>.csv.gz` | One row per test molecule per model: true and noisy labels, predicted mean, epistemic and aleatoric variance, similarity to the training set, replicate SD (AqSolDB) |
| `results/v2/metrics_all.csv` | All metric files merged by `scripts/collect.py` |
