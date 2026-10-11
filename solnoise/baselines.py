"""Classical baselines: a Gaussian process and a random forest."""

import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler


class GaussianProcessModel:
    """Gaussian process on standardized descriptors.

    The kernel has one length scale per descriptor and a white-noise term, so
    the model estimates the label noise from the data instead of being told it.
    """

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, x, y):
        self.x_scaler = StandardScaler().fit(x)
        self.y_mean, self.y_std = float(np.mean(y)), float(np.std(y))
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * RBF(length_scale=np.ones(x.shape[1]), length_scale_bounds=(1e-1, 1e3))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1e1))
        )
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, random_state=self.seed)
        with warnings.catch_warnings():
            # A length scale that reaches its upper bound means the model has decided
            # that descriptor does not matter. That is fine, so the warning is silenced.
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp.fit(self.x_scaler.transform(x), (y - self.y_mean) / self.y_std)
        return self

    def predict(self, x):
        mean, std = self.gp.predict(self.x_scaler.transform(x), return_std=True)
        return mean * self.y_std + self.y_mean, std * self.y_std


class RandomForestModel:
    """Random forest on Morgan fingerprints. Uncertainty is the spread across trees."""

    def __init__(self, seed=0, n_trees=300):
        self.forest = RandomForestRegressor(
            n_estimators=n_trees, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )

    def fit(self, x, y):
        self.forest.fit(x, y)
        return self

    def predict(self, x):
        per_tree = np.stack([tree.predict(x) for tree in self.forest.estimators_])
        return per_tree.mean(axis=0), per_tree.std(axis=0)
