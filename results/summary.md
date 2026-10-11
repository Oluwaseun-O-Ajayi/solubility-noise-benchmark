Mean ± standard deviation over 3 seeds. Test labels are always clean.

### Test RMSE (log10 mol/L, lower is better): random split

| Model | noise 0 | noise 0.3 | noise 0.6 | noise 1 |
|---|---|---|---|---|
| Training mean (no model) | 2.20 ± 0.14 | 2.20 ± 0.14 | 2.20 ± 0.15 | 2.20 ± 0.15 |
| Gaussian process (descriptors) | 0.64 ± 0.02 | 0.69 ± 0.03 | 0.70 ± 0.06 | 0.74 ± 0.06 |
| Random forest (fingerprints) | 1.15 ± 0.12 | 1.15 ± 0.12 | 1.17 ± 0.11 | 1.21 ± 0.10 |
| Fingerprint network (PyTorch) | 1.14 ± 0.10 | 1.15 ± 0.09 | 1.16 ± 0.09 | 1.23 ± 0.11 |
| Graph network (PyTorch) | 0.57 ± 0.03 | 0.59 ± 0.03 | 0.63 ± 0.06 | 0.72 ± 0.05 |

### Test RMSE (log10 mol/L, lower is better): scaffold split

| Model | noise 0 | noise 0.3 | noise 0.6 | noise 1 |
|---|---|---|---|---|
| Training mean (no model) | 3.10 ± 0.16 | 3.10 ± 0.17 | 3.11 ± 0.18 | 3.12 ± 0.18 |
| Gaussian process (descriptors) | 1.18 ± 0.08 | 1.06 ± 0.04 | 0.93 ± 0.15 | 0.88 ± 0.19 |
| Random forest (fingerprints) | 2.25 ± 0.12 | 2.26 ± 0.10 | 2.29 ± 0.08 | 2.31 ± 0.07 |
| Fingerprint network (PyTorch) | 2.02 ± 0.03 | 2.10 ± 0.03 | 2.10 ± 0.06 | 2.19 ± 0.19 |
| Graph network (PyTorch) | 0.72 ± 0.13 | 0.71 ± 0.09 | 0.73 ± 0.10 | 0.77 ± 0.07 |

### Coverage of the one-sigma interval (ideal 0.68): random split

| Model | noise 0 | noise 0.3 | noise 0.6 | noise 1 |
|---|---|---|---|---|
| Training mean (no model) | 0.66 ± 0.03 | 0.67 ± 0.02 | 0.68 ± 0.03 | 0.70 ± 0.03 |
| Gaussian process (descriptors) | 0.70 ± 0.02 | 0.73 ± 0.02 | 0.81 ± 0.02 | 0.91 ± 0.03 |
| Random forest (fingerprints) | 0.61 ± 0.04 | 0.64 ± 0.07 | 0.67 ± 0.06 | 0.73 ± 0.03 |
| Fingerprint network (PyTorch) | 0.68 ± 0.08 | 0.68 ± 0.07 | 0.73 ± 0.07 | 0.77 ± 0.04 |
| Graph network (PyTorch) | 0.74 ± 0.01 | 0.77 ± 0.04 | 0.82 ± 0.03 | 0.88 ± 0.02 |

### Coverage of the one-sigma interval (ideal 0.68): scaffold split

| Model | noise 0 | noise 0.3 | noise 0.6 | noise 1 |
|---|---|---|---|---|
| Training mean (no model) | 0.48 ± 0.07 | 0.49 ± 0.06 | 0.51 ± 0.07 | 0.53 ± 0.06 |
| Gaussian process (descriptors) | 0.52 ± 0.09 | 0.61 ± 0.08 | 0.76 ± 0.11 | 0.86 ± 0.08 |
| Random forest (fingerprints) | 0.41 ± 0.09 | 0.42 ± 0.06 | 0.44 ± 0.06 | 0.49 ± 0.04 |
| Fingerprint network (PyTorch) | 0.37 ± 0.04 | 0.38 ± 0.04 | 0.42 ± 0.04 | 0.43 ± 0.01 |
| Graph network (PyTorch) | 0.71 ± 0.04 | 0.78 ± 0.06 | 0.84 ± 0.05 | 0.91 ± 0.03 |

### Rank correlation between predicted uncertainty and actual error (higher is better): random split

| Model | noise 0 | noise 0.3 | noise 0.6 | noise 1 |
|---|---|---|---|---|
| Training mean (no model) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Gaussian process (descriptors) | 0.31 ± 0.06 | 0.29 ± 0.05 | 0.26 ± 0.13 | 0.26 ± 0.15 |
| Random forest (fingerprints) | 0.24 ± 0.09 | 0.23 ± 0.09 | 0.18 ± 0.12 | 0.10 ± 0.13 |
| Fingerprint network (PyTorch) | 0.14 ± 0.13 | 0.15 ± 0.11 | 0.11 ± 0.06 | 0.05 ± 0.10 |
| Graph network (PyTorch) | 0.47 ± 0.04 | 0.40 ± 0.05 | 0.35 ± 0.04 | 0.29 ± 0.06 |

### Rank correlation between predicted uncertainty and actual error (higher is better): scaffold split

| Model | noise 0 | noise 0.3 | noise 0.6 | noise 1 |
|---|---|---|---|---|
| Training mean (no model) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Gaussian process (descriptors) | 0.44 ± 0.07 | 0.43 ± 0.03 | 0.35 ± 0.07 | 0.20 ± 0.15 |
| Random forest (fingerprints) | -0.29 ± 0.04 | -0.29 ± 0.01 | -0.36 ± 0.03 | -0.36 ± 0.09 |
| Fingerprint network (PyTorch) | -0.28 ± 0.10 | -0.31 ± 0.06 | -0.18 ± 0.12 | -0.24 ± 0.10 |
| Graph network (PyTorch) | 0.18 ± 0.07 | 0.14 ± 0.17 | -0.01 ± 0.22 | 0.05 ± 0.11 |
