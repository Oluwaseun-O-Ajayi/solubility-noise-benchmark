"""Models with a common interface for the revised benchmark.

Every model has ``fit(x_train, y_train, x_valid, y_valid)`` and
``predict(x) -> (mean, var_epistemic, var_aleatoric)``. The total predictive
variance is the sum of the two parts. Keeping them separate lets the benchmark
score the interval for a new noisy measurement (total) and the interval for the
underlying property (epistemic only) from the same fitted model.
"""

import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

from .training import DeepEnsemble

# Columns of features.descriptor_matrix used by Delaney's ESOL equation:
# log P (Crippen here, ClogP in the original), molecular weight, rotatable bonds, aromatic proportion.
ESOL_COLUMNS = [1, 0, 6, 11]


def _subsample(n, max_n, seed):
    if max_n is None or n <= max_n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=max_n, replace=False))


class TrainMean:
    """Predict the mean training label; its spread is the uncertainty."""

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        self.mean, self.var, self.n = float(np.mean(y_train)), float(np.var(y_train)), len(y_train)
        return self

    def predict(self, x):
        n = len(x)
        return np.full(n, self.mean), np.full(n, self.var / self.n), np.full(n, self.var)


class ESOLLinear:
    """Ordinary least squares on the four descriptors of Delaney's ESOL equation.

    The aleatoric variance is the residual variance; the epistemic variance is the
    variance of the fitted mean, s^2 x (X^T X)^-1 x^T.
    """

    def __init__(self, seed=0):
        self.seed = seed

    def _design(self, x):
        return np.column_stack([np.ones(len(x)), np.asarray(x)[:, ESOL_COLUMNS]])

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        a = self._design(x_train)
        self.coef, *_ = np.linalg.lstsq(a, y_train, rcond=None)
        resid = y_train - a @ self.coef
        self.s2 = float(resid @ resid / (len(y_train) - a.shape[1]))
        self.xtx_inv = np.linalg.inv(a.T @ a)
        return self

    def predict(self, x):
        a = self._design(x)
        var_epi = self.s2 * np.einsum("ij,jk,ik->i", a, self.xtx_inv, a)
        return a @ self.coef, var_epi, np.full(len(x), self.s2)


class DescriptorGP:
    """Gaussian process on standardized descriptors: constant x ARD RBF + white noise.

    The white-noise variance is the aleatoric part. The posterior variance of the
    latent function is the epistemic part.
    """

    def __init__(self, seed=0, max_train=None):
        self.seed, self.max_train = seed, max_train

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        keep = _subsample(len(y_train), self.max_train, self.seed)
        x, y = np.asarray(x_train)[keep], np.asarray(y_train)[keep]
        self.x_scaler = StandardScaler().fit(x)
        self.y_mean, self.y_std = float(np.mean(y)), float(np.std(y))
        kernel = (ConstantKernel(1.0, (1e-2, 1e2))
                  * RBF(length_scale=np.ones(x.shape[1]), length_scale_bounds=(1e-1, 1e3))
                  + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1e1)))
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp.fit(self.x_scaler.transform(x), (y - self.y_mean) / self.y_std)
        self.noise_level = float(self.gp.kernel_.k2.noise_level)
        self.length_scales = np.asarray(self.gp.kernel_.k1.k2.length_scale)
        return self

    def predict(self, x):
        mean, std = self.gp.predict(self.x_scaler.transform(x), return_std=True)
        var_total = std ** 2
        var_epi = np.clip(var_total - self.noise_level, 0.0, None)
        scale = self.y_std ** 2
        return mean * self.y_std + self.y_mean, var_epi * scale, np.full(len(x), self.noise_level * scale)


def tanimoto(a, b):
    """Tanimoto similarity between rows of two binary matrices."""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    inter = a @ b.T
    denom = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where(denom > 0, inter / denom, 1.0)
    return sim.astype(np.float64)


class TanimotoGP:
    """Gaussian process on Morgan fingerprints with a Tanimoto kernel plus white noise.

    k(x, x') = a * T(x, x'), noise variance w. Both are fitted by maximizing the
    log marginal likelihood (analytic gradient, L-BFGS-B) on standardized labels.
    """

    def __init__(self, seed=0, max_train=None):
        self.seed, self.max_train = seed, max_train

    def _nlml(self, log_params, t, y):
        amp, noise = np.exp(log_params)
        k = amp * t
        k[np.diag_indices_from(k)] += noise
        try:
            c = cho_factor(k, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros(2)
        alpha = cho_solve(c, y, check_finite=False)
        nlml = 0.5 * y @ alpha + np.log(np.diag(c[0])).sum() + 0.5 * len(y) * np.log(2 * np.pi)
        k_inv = cho_solve(c, np.eye(len(y)), check_finite=False)
        inner = np.outer(alpha, alpha) - k_inv
        grad_amp = -0.5 * np.sum(inner * (amp * t))
        grad_noise = -0.5 * noise * np.trace(inner)
        return nlml, np.array([grad_amp, grad_noise])

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        keep = _subsample(len(y_train), self.max_train, self.seed)
        self.x = np.asarray(x_train)[keep]
        y = np.asarray(y_train, dtype=np.float64)[keep]
        self.y_mean, self.y_std = float(np.mean(y)), float(np.std(y))
        ys = (y - self.y_mean) / self.y_std
        t = tanimoto(self.x, self.x)
        bounds = [(np.log(1e-2), np.log(1e2)), (np.log(1e-4), np.log(1e1))]
        result = minimize(self._nlml, x0=np.log([1.0, 0.1]), args=(t, ys), jac=True,
                          method="L-BFGS-B", bounds=bounds)
        self.amp, self.noise_level = np.exp(result.x)
        k = self.amp * t
        k[np.diag_indices_from(k)] += self.noise_level
        self.chol = np.linalg.cholesky(k)
        self.alpha = cho_solve((self.chol, True), ys, check_finite=False)
        return self

    def predict(self, x, batch=2048):
        means, var_epi = [], []
        for start in range(0, len(x), batch):
            ks = self.amp * tanimoto(np.asarray(x)[start:start + batch], self.x)
            means.append(ks @ self.alpha)
            v = solve_triangular(self.chol, ks.T, lower=True, check_finite=False)
            var_epi.append(np.clip(self.amp - (v ** 2).sum(0), 0.0, None))
        scale = self.y_std ** 2
        mean = np.concatenate(means) * self.y_std + self.y_mean
        return mean, np.concatenate(var_epi) * scale, np.full(len(x), self.noise_level * scale)


class RandomForest:
    """Random forest; the spread of per-tree predictions is the (epistemic) uncertainty."""

    def __init__(self, seed=0, n_trees=300, min_samples_leaf=2, max_features=1.0):
        self.forest = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=min_samples_leaf,
                                            max_features=max_features, n_jobs=1, random_state=seed)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        self.forest.fit(x_train, y_train)
        return self

    def predict(self, x):
        per_tree = np.stack([tree.predict(x) for tree in self.forest.estimators_])
        return per_tree.mean(0), per_tree.var(0), np.zeros(len(x))


class Ensemble:
    """Deep ensemble wrapper that keeps the aleatoric and epistemic parts separate."""

    def __init__(self, kind, seed=0, n_members=5, **kwargs):
        self.ensemble = DeepEnsemble(kind, seed=seed, n_members=n_members, **kwargs)

    def fit(self, x_train, y_train, x_valid, y_valid):
        self.ensemble.fit(x_train, y_train, x_valid, y_valid)
        return self

    def predict(self, x):
        means, stds = zip(*(member.predict(x) for member in self.ensemble.members))
        means, variances = np.stack(means), np.stack(stds) ** 2
        return means.mean(0), means.var(0), variances.mean(0)


# name -> (representation, constructor)
REGISTRY = {
    "train_mean": ("descriptors", lambda seed, cfg: TrainMean(seed)),
    "esol_linear": ("descriptors", lambda seed, cfg: ESOLLinear(seed)),
    "gp_desc": ("descriptors", lambda seed, cfg: DescriptorGP(seed, **cfg)),
    "gp_tanimoto": ("fingerprints", lambda seed, cfg: TanimotoGP(seed, **cfg)),
    "rf_fp": ("fingerprints", lambda seed, cfg: RandomForest(seed, **cfg)),
    "rf_desc": ("descriptors", lambda seed, cfg: RandomForest(seed, **cfg)),
    "fp_net": ("fingerprints", lambda seed, cfg: Ensemble("fingerprint", seed=seed, **cfg)),
    "graph_net": ("graphs", lambda seed, cfg: Ensemble("graph", seed=seed, **cfg)),
}
MODEL_NAMES = list(REGISTRY)
