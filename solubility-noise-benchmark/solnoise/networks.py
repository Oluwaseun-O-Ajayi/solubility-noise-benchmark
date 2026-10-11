"""PyTorch models: a fingerprint network and a message-passing graph network.

Both networks predict a mean and a log-variance for each molecule, so each one
reports how noisy it thinks the answer is.
"""

import torch
from torch import nn

from .features import ATOM_DIM, BOND_DIM

LOGVAR_MIN, LOGVAR_MAX = -6.0, 3.0


class FingerprintNet(nn.Module):
    """Feed-forward network on Morgan fingerprints."""

    def __init__(self, n_inputs=2048, hidden=(512, 256), dropout=0.2):
        super().__init__()
        layers, width = [], n_inputs
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 2)

    def forward(self, x):
        out = self.head(self.body(x))
        return out[:, 0], out[:, 1].clamp(LOGVAR_MIN, LOGVAR_MAX)


class MessagePassingLayer(nn.Module):
    """One round of message passing.

    Each atom receives a message from every bonded neighbour. The message is
    computed from the neighbour's state and the bond between them. Messages are
    summed, then a GRU cell updates the atom's state.
    """

    def __init__(self, hidden):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(hidden + BOND_DIM, hidden), nn.ReLU())
        self.update = nn.GRUCell(hidden, hidden)

    def forward(self, h, edge_index, edge_attr):
        source, target = edge_index
        messages = self.message(torch.cat([h[source], edge_attr], dim=1))
        summed = torch.zeros_like(h).index_add_(0, target, messages)
        return self.update(summed, h)


class GraphNet(nn.Module):
    """Message-passing neural network (Gilmer et al., 2017) written in plain PyTorch."""

    def __init__(self, hidden=64, n_steps=3, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(ATOM_DIM, hidden)
        self.layers = nn.ModuleList([MessagePassingLayer(hidden) for _ in range(n_steps)])
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2)
        )

    def forward(self, batch):
        h = torch.relu(self.embed(batch.x))
        for layer in self.layers:
            h = layer(h, batch.edge_index, batch.edge_attr)
        # Pool atoms into one vector per molecule: the sum and the mean of atom states.
        total = torch.zeros(batch.n_graphs, h.shape[1]).index_add_(0, batch.batch, h)
        counts = torch.zeros(batch.n_graphs).index_add_(0, batch.batch, torch.ones(len(h)))
        pooled = torch.cat([total, total / counts.clamp(min=1).unsqueeze(1)], dim=1)
        out = self.readout(pooled)
        return out[:, 0], out[:, 1].clamp(LOGVAR_MIN, LOGVAR_MAX)


def gaussian_nll(mean, logvar, target):
    """Negative log-likelihood of ``target`` under N(mean, exp(logvar)), up to a constant."""
    return 0.5 * (logvar + (target - mean) ** 2 / logvar.exp()).mean()
