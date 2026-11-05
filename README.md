# Solubility Noise Benchmark

**Label noise and distribution shift in aqueous solubility models**

**This repository measures how much the accuracy and the reported uncertainty of molecular property models degrade when training labels carry error, and separates that effect from the effect of structural novelty and of a shift in the target distribution. Two data sets, three splits, two noise models, eight models, 300 conditions and 2,400 model fits.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23308184.svg)](https://doi.org/10.5281/zenodo.23308184)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## DOI and archival record

This software is archived on Zenodo for long-term preservation and citation.

**Current version:** 1.0.1

**DOI (all versions):** https://doi.org/10.5281/zenodo.23308184

This concept DOI always resolves to the most recent release. Each release also has its own version DOI, shown on its Zenodo record under "Versions".

Releases 1.0.x accompany the manuscript. Earlier releases (0.1.x) cover an ESOL-only study with four models and two splits, and their results are superseded.

---

## Overview

Experimental property data used to train molecular machine learning models contain measurement error, yet benchmarks usually treat reported values as exact. This repository asks four questions:

1. How much does accuracy degrade with label noise, and how does that compare with the effect of structural novelty and of a shift in the label distribution?
2. How does the coverage of predictive intervals change with label noise, and can the change be predicted?
3. Does predicted uncertainty track error on unfamiliar chemistry, and does any failure track the model or the input representation?
4. Does recalibration on a held-out set repair the intervals?

The benchmark provides:

- **Eight models**, arranged so that model class and input representation can be separated: Gaussian processes and random forests each on descriptors and on Morgan fingerprints, deep ensembles on fingerprints and on molecular graphs, and two baselines.
- **Two data sets**: ESOL (1,128 compounds) and a curated subset of AqSolDB (8,489 compounds).
- **Three splits**: random, Bemis-Murcko scaffold, and a label-matched Taylor-Butina cluster split that reproduces the structural novelty of a scaffold split without its shift in the target distribution.
- **Two noise models**: Gaussian at five levels and outlier corruption at three rates on ESOL; three and one on AqSolDB.
- **Uncertainty evaluation**: coverage at two nominal levels against both the reported labels and an independent noisy copy of them, negative log-likelihood, miscalibration area, error-based calibration with ENCE, rank correlation against a simulated reference, variance scaling and normalized split-conformal recalibration.
- **10 seeds per ESOL condition and 5 per AqSolDB condition**, reported as means with 95% bootstrap confidence intervals.

---

## Key findings

1. **Input representation changes accuracy more than model class.** Paired within seeds on clean labels, descriptors beat Morgan fingerprints for the same random forest by 0.31 to 0.84 log units across the six data set and split combinations. Over all noise levels the range is 0.25 to 0.85, and every one of the 36 paired intervals excludes zero. The same ordering holds for the Gaussian processes and the deep ensembles.
2. **Predictive intervals are calibrated for a new measurement, not for the reported value.** Coverage of the reported test labels rises well above nominal as training labels get noisier. Scored against an independent noisy copy of the test labels, 95% coverage stays between 0.917 and 0.938 at every noise level, and a closed-form expression reproduces the apparent excess.
3. **Distribution shift breaks uncertainty in a way recalibration does not repair.** Under scaffold splitting on ESOL, the fingerprint ensemble's uncertainty correlates negatively with its error, because its error stops falling as test molecules become more similar to the training set while its predicted uncertainty keeps falling. Variance scaling and conformal correction restore interval width but leave the ordering untouched.
4. **A Gaussian process recovers the amount of noise that was added**, to within 0.09 log units for sigma at or above 0.6, and the noise it fits to unaltered labels is an upper bound on the error in the data as supplied.
5. **Aleatoric and epistemic components are not comparable across model families.** Epistemic-only coverage spans 0.04 to 0.97 across models, which reflects how each model is parameterized rather than a quantity the models share.
6. **Measured inter-source disagreement predicts model error and models do not detect it.** For AqSolDB compounds reported by several sources, those with above-median disagreement are harder to predict, yet the aleatoric estimates of the four non-baseline models that report one track that disagreement with rank correlations of 0.02 to 0.25.

Full results are in `results/v2/`, and the figures that summarize them in `figures/v2/`.

---

## Scope and limitations

This is a controlled benchmark on one property, intended for research, education and methodological comparison. The models are not validated for prospective solubility prediction or for decision-critical use.

- One property, two data sets. Aqueous solubility has an unusually well-established descriptor baseline, so the representation result should not be read as a general claim.
- Added noise is independent Gaussian or outlier corruption. Real assay error can be systematic, method-dependent or correlated with structure.
- "Noise 0" means no added noise, not error-free labels. The reported values already carry measurement error.
- The independent noisy copy of the test labels is drawn from the same generator as the training noise, so agreement between the two is a consistency check. The AqSolDB replicate comparison partly closes that gap.
- The cluster split chooses, among 200 random cluster orderings, the one whose test-set mean label is closest to the training mean. That selection reads the test labels. No model is given a test label, but a model's absolute score under this split should not be quoted as held-out performance.
- On AqSolDB both Gaussian processes are fitted to a random subset of 4,000 training compounds, so AqSolDB comparisons involving a GP are confounded with training-set size.
- One configuration per model, selected on clean labels before any noise experiment and held across all conditions.
- Neural-network results use one PyTorch thread. Different thread counts change the order of floating-point operations and shift results slightly, within the seed-to-seed spread.

---

## Models

| Model | Input representation | Aleatoric variance | Epistemic variance |
|---|---|---|---|
| Training mean | None | Training-label variance | Variance of the mean |
| ESOL equation | Four Delaney descriptors | Residual variance | Variance of the fitted mean |
| GP, descriptors | 12 RDKit descriptors | Fitted white-noise variance | Posterior variance of the latent function |
| GP, Tanimoto | Morgan fingerprints (radius 2, 2,048 bits) | Fitted white-noise variance | Posterior variance of the latent function |
| RF, descriptors | 12 RDKit descriptors | None | Variance across trees |
| RF, fingerprints | Morgan fingerprints | None | Variance across trees |
| Ensemble, fingerprints | Morgan fingerprints | Mean of member variances | Variance of member means |
| Ensemble, graph | Atom and bond features | Mean of member variances | Variance of member means |

The graph network is a message-passing neural network (Gilmer et al., 2017) in plain PyTorch, without a graph library. Each atom receives messages from its bonded neighbours over three rounds, a GRU cell updates the atom state, and sum and mean pooling give one vector per molecule. Both networks predict a mean and a variance per molecule and are trained on a Gaussian negative log-likelihood after a 10-epoch squared-error warm-up, as five-member deep ensembles (Lakshminarayanan et al., 2017). The Tanimoto GP uses k(x, x') = a·T(x, x') plus white noise (Ralaivola et al., 2005), with the amplitude and noise fitted by maximum marginal likelihood using an analytic gradient.

---

## Experimental design

**Data.** ESOL is used as distributed with MoleculeNet. AqSolDB is verified against its SHA-256 checksum, then curated by removing multi-component entries, unparseable SMILES, entries containing elements outside common organic chemistry, molecules with fewer than two heavy atoms, and duplicates after canonicalization without stereochemistry, leaving 8,489 compounds.

**Splits.** All splits are 80/10/10. Random permutes by seed. Scaffold assigns whole Bemis-Murcko scaffold groups, largest first to training. Cluster groups molecules by Taylor-Butina clustering of Morgan fingerprints at a Tanimoto threshold of 0.4, assigns whole clusters, and keeps the ordering whose test mean label is closest to the training mean.

**Label noise.** Applied to training and validation labels only; test labels are never altered. Gaussian at sigma 0, 0.3, 0.6, 1.0 and 2.0 on ESOL, and 0, 0.6 and 1.0 on AqSolDB. Outlier corruption at 5%, 10% and 20% on ESOL and 10% on AqSolDB, each corrupted label shifted by 1 to 3 log units with random sign.

**Grid.** 240 ESOL conditions (3 splits x 8 noise settings x 10 seeds) and 60 AqSolDB conditions (3 splits x 4 noise settings x 5 seeds).

---

## Installation

Python 3.10 or later, CPU only.

```bash
git clone https://github.com/Oluwaseun-O-Ajayi/solubility-noise-benchmark.git
cd solubility-noise-benchmark
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

`requirements-lock.txt` records the exact versions used for the published run. To reproduce that environment rather than the latest compatible one, install from it instead.

---

## Usage

### Full grid

```bash
python scripts/make_grid.py                       # writes grids/esol.csv and grids/aqsoldb.csv
bash scripts/run_local.sh grids/esol.csv 4        # 4 worker processes
python scripts/collect.py                         # merge metrics, report what is missing
python scripts/analyze.py                         # summary tables, paired comparisons, figures
```

### One condition

```bash
python scripts/run_benchmark.py --dataset esol --split scaffold --noise gaussian \
    --level 1.0 --seed 0 --threads 1 --out results/v2
```

### On a Slurm cluster

```bash
sbatch slurm/test_one.sh          # one condition, plus the test suite
sbatch slurm/esol_array.sh        # 240 conditions
sbatch slurm/aqsoldb_array.sh     # 60 conditions
```

`docs/SAPELO2.md` gives a step-by-step walkthrough for the University of Georgia Sapelo2 cluster, which was used for the published run. The array scripts pass `--skip-existing`, so resubmitting the same array reruns only missing conditions.

---

## Project structure

```
solubility-noise-benchmark/
├── solnoise/
│   ├── data.py          # ESOL and AqSolDB loading, curation, checksum
│   ├── features.py      # Descriptors, Morgan fingerprints, molecular graphs
│   ├── splits.py        # Random, scaffold, Taylor-Butina cluster splits
│   ├── noise.py         # Gaussian and outlier label noise
│   ├── baselines.py     # Earlier GP and random forest wrappers
│   ├── models.py        # All eight models behind one fit/predict interface
│   ├── networks.py      # Fingerprint network, graph network, Gaussian likelihood
│   ├── training.py      # Training loop, early stopping, deep ensembles
│   ├── metrics.py       # RMSE, MAE, coverage, uncertainty-error correlation
│   ├── uq_metrics.py    # NLL, miscalibration area, ENCE, recalibration, conformal
│   ├── benchmark.py     # One condition: split, noise, seed, all models
│   └── experiment.py    # Earlier single-condition driver
├── scripts/
│   ├── make_grid.py     run_benchmark.py   run_local.sh
│   ├── collect.py       analyze.py         tune.py
│   ├── figures/         # Per-figure scripts
│   └── run_experiments.py  summarize.py    gp_diagnostic.py
├── configs/models.json  # One fixed configuration per model
├── grids/               # Condition lists for the full grid
├── slurm/               # Job array scripts
├── tests/               # 36 unit tests
├── data/                # esol.csv, aqsoldb.csv, provenance and licence notes
├── results/v2/          # metrics/, predictions/, merged tables, summary, paired
├── figures/             # Figure scripts and output
└── docs/SAPELO2.md
```

---

## Testing

36 tests covering featurization, split integrity (no scaffold or cluster shared between sets), noise reproducibility, uncertainty metrics against analytically known calibrated and miscalibrated cases, conformal coverage on exchangeable data, and model output shapes and variances. The graph network is checked for invariance to atom ordering and for giving the same prediction alone or in a batch.

```bash
pytest -q
```

---

## Output files

- `results/v2/metrics/<condition>.csv`: one row per model, all metrics for that condition
- `results/v2/predictions/<condition>.csv.gz`: one row per test molecule per model, with predicted mean, epistemic and aleatoric variance, maximum Tanimoto similarity to training, and recorded inter-source SD where available
- `results/v2/metrics_all.csv`: all metric files merged
- `results/v2/summary.csv`: means with 95% bootstrap confidence intervals
- `results/v2/paired.csv`: within-seed paired model differences
- `results/tuning/tuning.csv`: the hyperparameter search
- `figures/v2/`: all figures, PNG and PDF at 600 dpi

---

## Data sources and licences

- **ESOL**: Delaney, *J. Chem. Inf. Comput. Sci.*, 2004, **44**, 1000, as processed and distributed with MoleculeNet.
- **AqSolDB**: Sorkun, Khetan and Er, *Sci. Data*, 2019, **6**, 143, distributed under CC0. See `data/README.md`.

Code in this repository is MIT licensed.

---

## Citation

If you use this software, cite the archived version:

```bibtex
@software{ajayi_solubility_noise_benchmark_2026,
  author    = {Ajayi, Oluwaseun O.},
  title     = {Solubility Noise Benchmark: label noise and distribution shift in aqueous solubility models},
  version   = {1.0.1},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23308184},
  url       = {https://github.com/Oluwaseun-O-Ajayi/solubility-noise-benchmark}
}
```

---

## Author

Oluwaseun O. Ajayi, Department of Chemistry, University of Georgia.
ORCID: [0000-0003-0040-7217](https://orcid.org/0000-0003-0040-7217)
