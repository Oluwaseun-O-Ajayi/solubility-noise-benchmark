"""Choose one configuration per model before any noise experiment.

Selection uses clean labels, the random split of ESOL, seeds 0 to 2, and the
validation-set RMSE only (the test sets are never seen). Networks are tuned as
single members to keep the search cheap; the chosen settings are then used in
5-member ensembles. The winning settings are written to configs/models.json.

    python scripts/tune.py
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solnoise.benchmark import get_dataset  # noqa: E402
from solnoise.models import REGISTRY  # noqa: E402
from solnoise.splits import random_split  # noqa: E402

SEARCH = {
    "rf_fp": [{"n_trees": 300, "min_samples_leaf": l, "max_features": f}
              for l, f in itertools.product([1, 2, 5], [1.0, 0.33, "sqrt"])],
    "rf_desc": [{"n_trees": 300, "min_samples_leaf": l, "max_features": f}
                for l, f in itertools.product([1, 2, 5], [1.0, 0.33, "sqrt"])],
    "fp_net": [{"n_members": 1, "net_kwargs": {"hidden": list(h), "dropout": d}}
               for h, d in itertools.product([(256, 128), (512, 256), (1024, 256)], [0.1, 0.2, 0.4])],
    "graph_net": [{"n_members": 1, "net_kwargs": {"hidden": h, "n_steps": s, "dropout": d}}
                  for h, s, d in itertools.product([64, 128], [3, 4], [0.1, 0.2])],
}


def main():
    torch.set_num_threads(1)
    data = get_dataset("esol")
    out = ROOT / "results" / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, configs in SEARCH.items():
        kind, build = REGISTRY[name]
        x = data.representation(kind)
        for k, cfg in enumerate(configs):
            scores = []
            for seed in range(3):
                train, valid, _ = random_split(len(data.y), seed)
                model = build(seed, cfg).fit(x[train], data.y[train], x[valid], data.y[valid])
                mu, *_ = model.predict(x[valid])
                scores.append(float(np.sqrt(np.mean((mu - data.y[valid]) ** 2))))
            rows.append({"model": name, "config": json.dumps(cfg), "valid_rmse": np.mean(scores),
                         "valid_rmse_sd": np.std(scores)})
            print(f"{name} {k}: {np.mean(scores):.3f} {cfg}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "tuning.csv", index=False)

    config_path = ROOT / "configs" / "models.json"
    config = json.loads(config_path.read_text())
    for name in SEARCH:
        best = json.loads(table[table.model == name].sort_values("valid_rmse").iloc[0].config)
        if "n_members" in best:
            best["n_members"] = 5
        config["models"][name] = best
        print(f"selected {name}: {best}")
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
