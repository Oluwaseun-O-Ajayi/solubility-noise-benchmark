# How much measurement noise can a solubility model take?

I am a bench chemist. Every value I have measured came with error, and the public datasets that models learn from are built out of measurements like mine. This project asks a practical question: if the training labels are noisy, how much worse do the predictions get, and can the model's own uncertainty still be trusted?

I test four models on the ESOL aqueous solubility dataset (1,128 compounds), under two kinds of train/test split, with four levels of noise added to the training labels on purpose.

## Short answers

1. **Random noise costs little.** On a random split, adding noise with a standard deviation of 1.0 log unit (about half the spread of the whole dataset) raised the graph network's test error from 0.57 to 0.72.
2. **How you split the data matters more than the noise.** When the test set contains ring systems the model never saw (a scaffold split), the fingerprint models went from about 1.15 to above 2.0. No amount of label noise in this study did that much damage.
3. **Models trained on noisy labels become underconfident about the truth.** They learn the noise and include it in their error bars, so the error bars are too wide when judged against clean values.
4. **Fingerprint models fail silently on new scaffolds.** Their error bars covered the truth about 40% of the time instead of 68%, and they were more confident where they were more wrong.

## What is in the comparison

| Model | Input | Uncertainty comes from |
|---|---|---|
| Training mean | none | spread of the training labels |
| Gaussian process (scikit-learn) | 12 RDKit descriptors | the posterior, with a learned noise term |
| Random forest (scikit-learn) | Morgan fingerprints, 2048 bits | spread across 300 trees |
| Fingerprint network (PyTorch) | Morgan fingerprints, 2048 bits | deep ensemble of 5 networks |
| Graph network (PyTorch) | atoms and bonds | deep ensemble of 5 networks |

The graph network is a message-passing network written in plain PyTorch, with no graph library. Each atom collects messages from its bonded neighbours for three rounds, a GRU cell updates the atom's state, and the atom states are pooled into one vector per molecule. Both PyTorch models predict a mean and a variance for every molecule and are trained with a Gaussian likelihood.

**Splits.** A random split, and a scaffold split in which every Bemis-Murcko scaffold is kept whole inside train, validation, or test. Both are 80/10/10.

**Noise.** Gaussian noise with standard deviation 0, 0.3, 0.6, or 1.0 log10 mol/L is added to the training and validation labels. Validation labels get noise too, because in a real project they are measured the same way as the training labels. Test labels are never changed, so every score is distance from the reported value.

**Repeats.** Three random seeds. Tables show the mean ± standard deviation across seeds.

## Results

### Test error (RMSE in log10 mol/L, lower is better)

![Test error against label noise](figures/rmse_vs_noise.png)

Random split:

| Model | noise 0 | noise 0.3 | noise 0.6 | noise 1.0 |
|---|---|---|---|---|
| Training mean (no model) | 2.20 ± 0.14 | 2.20 ± 0.14 | 2.20 ± 0.15 | 2.20 ± 0.15 |
| Gaussian process | 0.64 ± 0.02 | 0.69 ± 0.03 | 0.70 ± 0.06 | 0.74 ± 0.06 |
| Random forest | 1.15 ± 0.12 | 1.15 ± 0.12 | 1.17 ± 0.11 | 1.21 ± 0.10 |
| Fingerprint network | 1.14 ± 0.10 | 1.15 ± 0.09 | 1.16 ± 0.09 | 1.23 ± 0.11 |
| Graph network | 0.57 ± 0.03 | 0.59 ± 0.03 | 0.63 ± 0.06 | 0.72 ± 0.05 |

Scaffold split:

| Model | noise 0 | noise 0.3 | noise 0.6 | noise 1.0 |
|---|---|---|---|---|
| Training mean (no model) | 3.10 ± 0.16 | 3.10 ± 0.17 | 3.11 ± 0.18 | 3.12 ± 0.18 |
| Gaussian process | 1.18 ± 0.08 | 1.06 ± 0.04 | 0.93 ± 0.15 | 0.88 ± 0.19 |
| Random forest | 2.25 ± 0.12 | 2.26 ± 0.10 | 2.29 ± 0.08 | 2.31 ± 0.07 |
| Fingerprint network | 2.02 ± 0.03 | 2.10 ± 0.03 | 2.10 ± 0.06 | 2.19 ± 0.19 |
| Graph network | 0.72 ± 0.13 | 0.71 ± 0.09 | 0.73 ± 0.10 | 0.77 ± 0.07 |

The scaffold split is hard on this dataset for a reason beyond new ring systems. The two largest scaffold groups (molecules with no rings, and plain benzene rings) go to training, so the test set is left with the more complex, less soluble molecules. Across the three seeds the average training label was about -2.9 and the average test label was between -4.3 and -5.1. The models are being asked to extrapolate, and the fingerprint models cannot.

### Can the error bars be trusted?

Coverage is the fraction of test molecules whose reported value falls inside the model's one-sigma interval. A well-calibrated model scores about 0.68. Lower means overconfident.

![One-sigma coverage against label noise](figures/coverage_vs_noise.png)

| Model | random, noise 0 | random, noise 1.0 | scaffold, noise 0 | scaffold, noise 1.0 |
|---|---|---|---|---|
| Gaussian process | 0.70 ± 0.02 | 0.91 ± 0.03 | 0.52 ± 0.09 | 0.86 ± 0.08 |
| Random forest | 0.61 ± 0.04 | 0.73 ± 0.03 | 0.41 ± 0.09 | 0.49 ± 0.04 |
| Fingerprint network | 0.68 ± 0.08 | 0.77 ± 0.04 | 0.37 ± 0.04 | 0.43 ± 0.01 |
| Graph network | 0.74 ± 0.01 | 0.88 ± 0.02 | 0.71 ± 0.04 | 0.91 ± 0.03 |

Two things stand out.

On the random split with clean labels, all four models are close to 0.68. As noise goes up, coverage climbs toward 0.9 for the Gaussian process and the graph network. They are doing what they were built to do: they estimate the noise in their training labels and report it. But the test labels here are clean, so against the truth those intervals are too wide. If you train on noisy assay data and care about the true value, the reported uncertainty overstates your error.

On the scaffold split, the random forest and the fingerprint network cover the truth only about 40% of the time, and the rank correlation between their uncertainty and their actual error is negative (about -0.3, see `results/summary.md`). They are most confident on the molecules they get most wrong. The graph network stays near 0.71 with clean labels.

### One result that runs against intuition

On the scaffold split, the Gaussian process got better as the labels got noisier: 1.18 with clean labels, 0.88 with noise of 1.0.

I checked what the kernel learned (`scripts/gp_diagnostic.py`, output in `results/gp_diagnostic.csv`). With clean labels the median length scale was 2.0 to 2.4. With noisy labels it was 5.8 to 28.7, and the learned noise term rose from about 0.04 to about 0.27 (in standardized units). In other words, noisy labels pushed the model toward a smoother function, and a smoother function extrapolated better to scaffolds it had not seen. The clean-label model was fitting fine detail that did not carry over.

This is three seeds on one dataset, so I read it as an observation worth testing further, not a rule.

## Limits

- One dataset of 1,128 compounds and three seeds. The standard deviations in the tables are from three runs.
- The ESOL values already contain measurement error, so "noise 0" means no added noise, not noise-free.
- The added noise is Gaussian and independent of the molecule. Real assay error can be systematic, and it can depend on the solubility range.
- No hyperparameter search. Every model uses one fixed, reasonable setting.
- Small models trained on a CPU.
- The results in this repository were run with one PyTorch thread (`--threads 1`). With a different thread count the network results shift slightly, because floating-point sums are added in a different order. In a spot check the graph network moved from 0.54 to 0.57 on one condition, which is inside the seed-to-seed spread.

## Run it yourself

```bash
git clone https://github.com/Oluwaseun-O-Ajayi/solubility-noise-benchmark.git
cd solubility-noise-benchmark
pip install -r requirements.txt

pytest -q                                       # 21 tests, about 15 seconds
python scripts/run_experiments.py --threads 1   # full grid, writes results/results.csv
python scripts/summarize.py                     # tables in results/summary.md, figures in figures/
python scripts/gp_diagnostic.py                 # the Gaussian process check
```

The full grid is 24 conditions. One condition takes about 4 minutes on a 2-core CPU, so allow about an hour and a half, or run each seed as its own process. For a quick look, run one condition:

```bash
python scripts/run_experiments.py --seeds 0 --splits random --sigmas 0 --out results/quick.csv
```

## Layout

```
solnoise/
  data.py         load ESOL and canonicalize the SMILES
  features.py     descriptors, Morgan fingerprints, and molecular graphs
  splits.py       random and scaffold splits
  noise.py        add Gaussian noise to labels
  baselines.py    Gaussian process and random forest
  networks.py     the two PyTorch models and the Gaussian likelihood
  training.py     training loop, early stopping, deep ensembles
  metrics.py      RMSE, MAE, coverage, uncertainty-error rank correlation
  experiment.py   one condition: split, noise level, seed, all models
scripts/          run the grid, summarize it, and the Gaussian process check
tests/            21 tests, including a check that the graph network gives the
                  same answer when a molecule's atoms are listed in a different order
data/esol.csv     the dataset
results/          raw results for every run, and summary tables
figures/          the plots above
```

## What I would do next

- Repeat on a second property. Lipophilicity (about 4,200 compounds) is the obvious one.
- Replace Gaussian noise with systematic error, such as an offset between two laboratories.
- Test whether a simple recalibration step fixes the too-wide intervals under noise.

## Data and references

`data/esol.csv` is the processed ESOL file distributed with DeepChem and MoleculeNet.

- Delaney, J. S. (2004). ESOL: Estimating aqueous solubility directly from molecular structure. *J. Chem. Inf. Comput. Sci.*, 44(3), 1000-1005.
- Wu, Z., et al. (2018). MoleculeNet: a benchmark for molecular machine learning. *Chem. Sci.*, 9, 513-530.
- Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks. *J. Med. Chem.*, 39(15), 2887-2893.
- Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. *J. Chem. Inf. Model.*, 50(5), 742-754.
- Gilmer, J., et al. (2017). Neural message passing for quantum chemistry. *ICML*.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.

## License

MIT. See `LICENSE`.
