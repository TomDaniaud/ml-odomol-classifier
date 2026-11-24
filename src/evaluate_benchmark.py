"""Évalue un ou plusieurs runs de validation croisée sur le jeu de test GS-LF, protocole Han et al. (2026).

Chaque run (`src/cross_val.py` avec `experiment=benchmark_pom`) produit `oof_predictions.pt`. Ce script :
  1. aligne les prédictions out-of-fold (OOF) de tous les runs par molécule (`source_index`) et en fait la moyenne ;
  2. apprend un seuil par label sur ces OOF (F1 maximal) ;
  3. fait la moyenne des probabilités de test de **tous** les modèles (5 plis × nombre de runs) ;
  4. calcule AUROC, AUPRC, F1, précision et rappel macro avec IC 95 % bootstrap (1 000 rééchantillonnages).

Avec un seul run, l'ensemble compte 5 modèles. L'article en utilise 50 (10 meilleures configurations × 5 plis) :
passer plusieurs runs permet de s'en rapprocher.

Usage :
    uv run src/evaluate_benchmark.py outputs/<date>/<heure>/ [outputs/... ...] [--out resultats.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rootutils
import torch

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.models.benchmark_metrics import evaluate_benchmark, format_table  # noqa: E402


def load_run(run_dir: Path) -> dict:
    payload = torch.load(run_dir / "oof_predictions.pt", map_location="cpu", weights_only=False)
    if not payload.get("fold_test_probs"):
        raise SystemExit(f"{run_dir} : pas de prédictions de test. Lancer cross_val.py avec experiment=benchmark_pom.")
    folds = sorted(payload["fold_probs"])
    return {
        "oof_source": np.concatenate([np.asarray(payload["fold_val_source_indices"][k]) for k in folds]),
        "oof_probs": torch.cat([payload["fold_probs"][k] for k in folds]).numpy(),
        "oof_targets": torch.cat([payload["fold_targets"][k] for k in folds]).numpy(),
        "test_source": np.asarray(payload["test_source_indices"]),
        "test_targets": payload["test_targets"].numpy(),
        "test_probs": [payload["fold_test_probs"][k].numpy() for k in sorted(payload["fold_test_probs"])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="dossiers de sortie de cross_val.py")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    runs = [load_run(r) for r in args.runs]
    ref = runs[0]

    # Alignement par molécule : tous les runs doivent partager le même fichier de split
    oof_order = np.argsort(ref["oof_source"])
    test_order = np.argsort(ref["test_source"])
    oof_probs, test_probs = [], []
    for run, path in zip(runs, args.runs, strict=True):
        if set(run["oof_source"]) != set(ref["oof_source"]) or set(run["test_source"]) != set(ref["test_source"]):
            raise SystemExit(f"{path} n'utilise pas le même découpage que {args.runs[0]}.")
        o = np.argsort(run["oof_source"])
        t = np.argsort(run["test_source"])
        oof_probs.append(run["oof_probs"][o])
        test_probs.extend(p[t] for p in run["test_probs"])

    result = evaluate_benchmark(
        oof_probs=np.mean(oof_probs, axis=0),
        oof_targets=ref["oof_targets"][oof_order],
        test_probs_per_model=test_probs,
        test_targets=ref["test_targets"][test_order],
        n_resamples=args.resamples,
        seed=args.seed,
    )
    result["runs"] = [str(r) for r in args.runs]
    print(format_table(result))
    out = args.out or (args.runs[0] / "benchmark_test_metrics_ensemble.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Écrit : {out}")


if __name__ == "__main__":
    main()
