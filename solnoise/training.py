"""Train the PyTorch models, and combine several of them into a deep ensemble."""

import copy

import numpy as np
import torch

from .features import collate_graphs
from .networks import FingerprintNet, GraphNet, gaussian_nll


def _batches(n, batch_size, rng):
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


class TorchRegressor:
    """One network trained with Adam, a Gaussian likelihood, and early stopping.

    ``kind`` is "fingerprint" (inputs are a 2-D array) or "graph" (inputs are a
    list of MolGraph objects). Targets are standardized with the training mean
    and standard deviation, then predictions are converted back.
    """

    def __init__(self, kind, seed=0, max_epochs=150, patience=20, batch_size=64,
                 lr=1e-3, weight_decay=1e-5, warmup_epochs=10):
        self.kind, self.seed = kind, seed
        self.max_epochs, self.patience, self.batch_size = max_epochs, patience, batch_size
        self.lr, self.weight_decay, self.warmup_epochs = lr, weight_decay, warmup_epochs
        self.history = []

    def _inputs(self, x, index):
        if self.kind == "graph":
            return collate_graphs([x[i] for i in index])
        return torch.as_tensor(x[index], dtype=torch.float32)

    def _forward_all(self, x, batch_size=256):
        means, logvars = [], []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                index = np.arange(start, min(start + batch_size, len(x)))
                mean, logvar = self.model(self._inputs(x, index))
                means.append(mean)
                logvars.append(logvar)
        return torch.cat(means), torch.cat(logvars)

    def fit(self, x_train, y_train, x_valid, y_valid):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.y_mean, self.y_std = float(np.mean(y_train)), float(np.std(y_train))
        t_train = torch.as_tensor((y_train - self.y_mean) / self.y_std, dtype=torch.float32)
        t_valid = torch.as_tensor((y_valid - self.y_mean) / self.y_std, dtype=torch.float32)

        self.model = GraphNet() if self.kind == "graph" else FingerprintNet(x_train.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss, best_state, stale = float("inf"), None, 0
        for epoch in range(self.max_epochs):
            self.model.train()
            for index in _batches(len(t_train), self.batch_size, rng):
                mean, logvar = self.model(self._inputs(x_train, index))
                if epoch < self.warmup_epochs:
                    # Fit the mean first. Learning the variance from the start can let
                    # the network explain every error away as "noise".
                    loss = ((mean - t_train[index]) ** 2).mean()
                else:
                    loss = gaussian_nll(mean, logvar, t_train[index])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

            mean, logvar = self._forward_all(x_valid)
            valid_mse = float(((mean - t_valid) ** 2).mean())
            valid_loss = valid_mse if epoch < self.warmup_epochs else float(gaussian_nll(mean, logvar, t_valid))
            self.history.append({"epoch": epoch, "valid_mse": valid_mse, "valid_loss": valid_loss})
            if epoch < self.warmup_epochs:
                continue
            if valid_loss < best_loss - 1e-4:
                best_loss, best_state, stale = valid_loss, copy.deepcopy(self.model.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, x):
        mean, logvar = self._forward_all(x)
        return (mean.numpy() * self.y_std + self.y_mean,
                np.sqrt(np.exp(logvar.numpy())) * self.y_std)


class DeepEnsemble:
    """Several networks that differ only in their random seed (Lakshminarayanan et al., 2017).

    The ensemble prediction is the average of the member means. Its variance has
    two parts: the average noise each member predicts, and how much the members
    disagree with one another.
    """

    def __init__(self, kind, seed=0, n_members=5, **train_kwargs):
        self.members = [TorchRegressor(kind, seed=1000 * seed + m, **train_kwargs) for m in range(n_members)]

    def fit(self, x_train, y_train, x_valid, y_valid):
        for member in self.members:
            member.fit(x_train, y_train, x_valid, y_valid)
        return self

    def predict(self, x):
        means, stds = zip(*(member.predict(x) for member in self.members))
        means, variances = np.stack(means), np.stack(stds) ** 2
        mean = means.mean(axis=0)
        variance = variances.mean(axis=0) + means.var(axis=0)
        return mean, np.sqrt(variance)
