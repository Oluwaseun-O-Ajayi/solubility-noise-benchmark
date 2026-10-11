import numpy as np
import torch
from rdkit import Chem

from solnoise.features import collate_graphs, fingerprint_matrix, mol_to_graph, smiles_to_graphs
from solnoise.networks import FingerprintNet, GraphNet, gaussian_nll
from solnoise.training import DeepEnsemble, TorchRegressor

SMILES = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "CCCCCC", "c1ccc2ccccc2c1", "CC(C)O",
          "OCC(O)CO", "Clc1ccccc1", "CCN(CC)CC", "O=C(O)c1ccccc1"]
LOGS = np.array([1.1, 0.0, -1.7, -3.8, -3.6, 0.4, 1.1, -2.4, 0.1, -1.6])


def test_network_output_shapes():
    graph_net, fp_net = GraphNet().eval(), FingerprintNet(n_inputs=256).eval()
    mean, logvar = graph_net(collate_graphs(smiles_to_graphs(SMILES)))
    assert mean.shape == logvar.shape == (len(SMILES),)
    mean, logvar = fp_net(torch.zeros(7, 256))
    assert mean.shape == logvar.shape == (7,)


def test_graph_net_ignores_atom_numbering():
    """The same molecule with its atoms listed in a different order must get the same answer."""
    torch.manual_seed(0)
    net = GraphNet().eval()
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    order = list(np.random.default_rng(0).permutation(mol.GetNumAtoms()))
    shuffled = Chem.RenumberAtoms(mol, [int(i) for i in order])
    with torch.no_grad():
        a, _ = net(collate_graphs([mol_to_graph(mol)]))
        b, _ = net(collate_graphs([mol_to_graph(shuffled)]))
    assert torch.allclose(a, b, atol=1e-5)


def test_graph_net_gives_same_answer_alone_or_in_a_batch():
    torch.manual_seed(0)
    net = GraphNet().eval()
    graphs = smiles_to_graphs(SMILES)
    with torch.no_grad():
        together, _ = net(collate_graphs(graphs))
        alone = torch.cat([net(collate_graphs([g]))[0] for g in graphs])
    assert torch.allclose(together, alone, atol=1e-5)


def test_gaussian_nll_prefers_the_right_variance():
    target = torch.tensor([0.0, 2.0, -2.0, 0.0])
    mean = torch.zeros(4)
    right = gaussian_nll(mean, torch.log(torch.full((4,), 2.0)), target)   # true mean square error is 2
    too_small = gaussian_nll(mean, torch.log(torch.full((4,), 0.05)), target)
    too_large = gaussian_nll(mean, torch.log(torch.full((4,), 50.0)), target)
    assert right < too_small and right < too_large


def test_training_lowers_validation_error():
    x = fingerprint_matrix(SMILES, n_bits=256)
    model = TorchRegressor("fingerprint", seed=0, max_epochs=60, patience=60, warmup_epochs=60)
    model.fit(x, LOGS, x, LOGS)
    assert model.history[-1]["valid_mse"] < 0.5 * model.history[0]["valid_mse"]


def test_ensemble_returns_positive_uncertainty():
    graphs = np.empty(len(SMILES), dtype=object)
    graphs[:] = smiles_to_graphs(SMILES)
    model = DeepEnsemble("graph", seed=0, n_members=2, max_epochs=15, patience=15, warmup_epochs=5)
    model.fit(graphs, LOGS, graphs, LOGS)
    mean, std = model.predict(graphs)
    assert mean.shape == std.shape == (len(SMILES),)
    assert np.all(std > 0) and np.all(np.isfinite(mean))
