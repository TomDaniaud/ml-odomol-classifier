"""Génère le découpage train / test + 5 plis du protocole POM / OpenPOM / Uni-Mol2 (Han et al., 2026).

Protocole reproduit (cf. OpenPOM `examples/ensemble_benchmark.ipynb` et `openpom/utils/data_utils.py`) :
  1. Jeu de test tenu à l'écart : stratification itérative multi-label d'ordre 2 (paires de labels),
     80 % train / 20 % test, sur les 4 983 molécules du CSV GS-LF (test ≈ 996 molécules).
  2. Validation croisée en 5 plis sur la partie train, même stratification itérative d'ordre 2.

Le fichier produit (JSON) contient des `source_index` (numéros de ligne du CSV) : il est indépendant
du fichier de conformères et fixe une fois pour toutes les molécules de test.

Usage :
    uv run src/scripts/make_benchmark_split.py                      # scikit-multilearn (ordre 2) si installé
    uv run src/scripts/make_benchmark_split.py --method iterstrat   # repli : iterstrat (ordre 1)
"""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "data" / "smiles_odor_dataset.csv"
DEFAULT_OUT = ROOT / "data" / "splits" / "gslf_pom_benchmark_split.json"
NON_LABEL_COLUMNS = {"nonStereoSMILES", "descriptors"}


def split_skmultilearn(y: np.ndarray, test_frac: float, n_folds: int, order: int, seed: int):
    """Reproduit `IterativeStratifiedSplitter(order=2).train_test_split` puis `.k_fold_split` d'OpenPOM."""
    from skmultilearn.model_selection import IterativeStratification

    # skmultilearn départage les ex-aequo avec le générateur global de NumPy : on le fixe
    # pour rendre le découpage reproductible (OpenPOM ne le fixait pas et sauvegardait le split).
    np.random.seed(seed)
    x = np.zeros((len(y), 1))
    stratifier = IterativeStratification(
        n_splits=2, order=order, sample_distribution_per_fold=[test_frac, 1.0 - test_frac]
    )
    train_idx, test_idx = next(stratifier.split(x, y))

    np.random.seed(seed + 1)
    kfold = IterativeStratification(n_splits=n_folds, order=order)
    folds = [train_idx[val_pos] for _, val_pos in kfold.split(x[train_idx], y[train_idx])]
    return train_idx, test_idx, folds


def split_iterstrat(y: np.ndarray, test_frac: float, n_folds: int, seed: int):
    """Repli sans dépendance supplémentaire : stratification itérative d'ordre 1 (Sechidis et al., 2011)."""
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

    x = np.zeros((len(y), 1))
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_idx, test_idx = next(msss.split(x, y))
    mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [train_idx[val_pos] for _, val_pos in mskf.split(x[train_idx], y[train_idx])]
    return train_idx, test_idx, folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--method", choices=["skmultilearn", "iterstrat"], default="skmultilearn")
    parser.add_argument("--order", type=int, default=2, help="ordre de stratification (skmultilearn uniquement)")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    label_names = [c for c in df.columns if c not in NON_LABEL_COLUMNS]
    y = df[label_names].to_numpy(dtype=np.int64)

    if args.method == "skmultilearn":
        try:
            train_idx, test_idx, folds = split_skmultilearn(y, args.test_frac, args.folds, args.order, args.seed)
        except ImportError as exc:
            raise SystemExit(
                "scikit-multilearn n'est pas installé. Installez-le (`uv add scikit-multilearn`) pour reproduire "
                "exactement le protocole POM (ordre 2), ou relancez avec `--method iterstrat` (ordre 1)."
            ) from exc
        order = args.order
    else:
        train_idx, test_idx, folds = split_iterstrat(y, args.test_frac, args.folds, args.seed)
        order = 1

    # Contrôles d'intégrité
    assert len(set(train_idx) & set(test_idx)) == 0
    fold_union = np.concatenate(folds)
    assert len(fold_union) == len(set(fold_union)) == len(train_idx)
    assert set(fold_union) == set(train_idx)

    test_pos = y[test_idx].sum(axis=0)
    payload = {
        "description": "GS-LF benchmark split (protocole POM / OpenPOM / Han et al. 2026)",
        "created": date.today().isoformat(),
        "method": args.method,
        "stratification_order": order,
        "seed": args.seed,
        "test_fraction": args.test_frac,
        "csv": args.csv.name,
        "csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "n_total": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "label_names": label_names,
        "test": sorted(int(i) for i in test_idx),
        "folds": [sorted(int(i) for i in fold) for fold in folds],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(f"Méthode : {args.method} (ordre {order}), seed {args.seed}")
    print(f"Train : {len(train_idx)} · Test : {len(test_idx)} · Plis : {[len(f) for f in folds]}")
    print(f"Positifs par label dans le test : min {int(test_pos.min())}, médiane {int(np.median(test_pos))}, "
          f"max {int(test_pos.max())} · labels absents du test : {int((test_pos == 0).sum())}")
    print(f"Écrit : {args.out}")


if __name__ == "__main__":
    main()
