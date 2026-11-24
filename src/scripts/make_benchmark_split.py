"""Découpage GS-LF du protocole POM / OpenPOM / Han et al. 2026 : 20 % de test tenu à l'écart, puis 5 plis
sur le reste, par stratification itérative d'ordre 2 (comme `IterativeStratifiedSplitter(order=2)` d'OpenPOM).
Les indices sont les lignes du CSV. Usage : uv run src/scripts/make_benchmark_split.py"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from skmultilearn.model_selection import IterativeStratification

ROOT = Path(__file__).resolve().parents[2]

y = pd.read_csv(ROOT / "data/smiles_odor_dataset.csv").drop(columns=["nonStereoSMILES", "descriptors"]).to_numpy()
x = np.zeros((len(y), 1))

np.random.seed(42)  # skmultilearn départage les ex-aequo avec le générateur global de NumPy
train, test = next(IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.2, 0.8]).split(x, y))
np.random.seed(43)
folds = [train[val] for _, val in IterativeStratification(n_splits=5, order=2).split(x[train], y[train])]

out = ROOT / "data/splits/gslf_pom_benchmark_split.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"test": sorted(test.tolist()), "folds": [sorted(f.tolist()) for f in folds]}))
print(f"Train {len(train)} (plis {[len(f) for f in folds]}) · Test {len(test)} -> {out}")
