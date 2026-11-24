"""Évaluation alignée sur Han et al. (2026), « A General-Purpose Molecular Foundation Model Transfers
Across Diverse Olfactory Tasks » (arXiv:2608.25893), tableau 1 et tableau S1.

Protocole de l'article, reproduit ici :
  - jeu de test GS-LF tenu à l'écart, jamais utilisé pendant l'entraînement ni pour régler quoi que ce soit ;
  - prédictions de test = moyenne des probabilités des modèles de l'ensemble (un modèle par pli) ;
  - seuils de décision **un par label**, appris sur les prédictions out-of-fold (OOF) du train, puis appliqués
    tels quels au test ;
  - métriques macro (moyenne non pondérée sur les labels) : AUROC, AUPRC, F1, précision, rappel ;
  - intervalle de confiance à 95 % : 1 000 rééchantillonnages bootstrap du jeu de test (percentiles 2,5 / 97,5).

Points que l'article ne précise pas, et choix faits ici :
  - critère d'optimisation des seuils : F1 maximal par label sur les prédictions OOF ;
  - AUPRC : `average_precision_score` de scikit-learn ;
  - labels sans positif dans un échantillon bootstrap : exclus des moyennes AUROC / AUPRC (non définies),
    comptés à 0 pour la précision, le rappel et le F1 (`zero_division=0`).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

# Valeurs publiées (tableau 1 et S1 de Han et al. 2026), même jeu GS-LF et même protocole.
PAPER_REFERENCE: dict[str, dict[str, float]] = {
    "OpenPOM": {"auroc": 0.8883, "auprc": 0.3748, "f1": 0.3748, "precision": 0.4060, "recall": 0.4049},
    "POM (Lee et al. 2023)": {"auroc": 0.8940, "f1": 0.3600, "precision": 0.3790},
    "Uni-Mol2 FT 84M": {"auroc": 0.8990, "auprc": 0.4077, "f1": 0.3991, "precision": 0.4288, "recall": 0.4291},
}
METRIC_NAMES = ("auroc", "auprc", "f1", "precision", "recall")


def fit_label_thresholds(oof_probs: np.ndarray, oof_targets: np.ndarray, default: float = 0.5) -> np.ndarray:
    """Seuil par label maximisant le F1 sur les prédictions out-of-fold (recherche exacte, pas de grille)."""
    n_labels = oof_targets.shape[1]
    thresholds = np.full(n_labels, default, dtype=np.float64)
    for j in range(n_labels):
        y = oof_targets[:, j]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        precision, recall, thr = precision_recall_curve(y, oof_probs[:, j])
        # precision/recall ont un point de plus que thr (le point final précision=1, rappel=0)
        p, r = precision[:-1], recall[:-1]
        f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
        thresholds[j] = float(thr[int(np.argmax(f1))])
    return thresholds


def _per_label_counts(preds: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = (preds * targets).sum(axis=0)
    fp = (preds * (1 - targets)).sum(axis=0)
    fn = ((1 - preds) * targets).sum(axis=0)
    return tp, fp, fn


def macro_metrics(probs: np.ndarray, targets: np.ndarray, thresholds: np.ndarray) -> dict[str, float]:
    """AUROC, AUPRC, F1, précision et rappel, en moyenne macro sur les labels."""
    targets = targets.astype(np.int64)
    pos = targets.sum(axis=0)
    defined = (pos > 0) & (pos < len(targets))  # AUROC / AUPRC définies seulement si les 2 classes existent

    aurocs = [roc_auc_score(targets[:, j], probs[:, j]) for j in np.flatnonzero(defined)]
    auprcs = [average_precision_score(targets[:, j], probs[:, j]) for j in np.flatnonzero(defined)]

    preds = (probs >= thresholds[None, :]).astype(np.int64)
    tp, fp, fn = _per_label_counts(preds, targets)
    precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)

    return {
        "auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "auprc": float(np.mean(auprcs)) if auprcs else float("nan"),
        "f1": float(f1.mean()),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
    }


def bootstrap_ci(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Intervalle de confiance par bootstrap (rééchantillonnage des molécules du test, avec remise)."""
    rng = np.random.default_rng(seed)
    n = len(targets)
    samples: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        res = macro_metrics(probs[idx], targets[idx], thresholds)
        for m in METRIC_NAMES:
            samples[m].append(res[m])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {m: (float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))) for m, v in samples.items()}


def evaluate_benchmark(
    oof_probs: np.ndarray,
    oof_targets: np.ndarray,
    test_probs_per_model: list[np.ndarray],
    test_targets: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict:
    """Pipeline complet de l'article : seuils OOF → ensemble sur le test → métriques + IC bootstrap."""
    thresholds = fit_label_thresholds(oof_probs, oof_targets)
    test_probs = np.mean(np.stack(test_probs_per_model, axis=0), axis=0)
    point = macro_metrics(test_probs, test_targets, thresholds)
    ci = bootstrap_ci(test_probs, test_targets, thresholds, n_resamples=n_resamples, seed=seed)
    return {
        "n_models": len(test_probs_per_model),
        "n_test": int(len(test_targets)),
        "n_oof": int(len(oof_targets)),
        "test": point,
        "test_ci95": ci,
        "label_thresholds": thresholds.tolist(),
        # Pour information : métriques OOF (train, validation croisée) avec les mêmes seuils (optimistes)
        "oof": macro_metrics(oof_probs, oof_targets, thresholds),
    }


def format_table(result: dict, model_name: str = "Odomol (ce projet)") -> str:
    """Tableau texte au format du tableau 1 de l'article."""
    header = f"{'Modèle':<26}" + "".join(f"{m.upper():>18}" for m in METRIC_NAMES)
    lines = [header, "-" * len(header)]
    for name, ref in PAPER_REFERENCE.items():
        lines.append(f"{name:<26}" + "".join(f"{ref[m]:>18.4f}" if m in ref else f"{'–':>18}" for m in METRIC_NAMES))
    pt, ci = result["test"], result["test_ci95"]
    lines.append(f"{model_name:<26}" + "".join(f"{pt[m]:>18.4f}" for m in METRIC_NAMES))
    lines.append(f"{'':<26}" + "".join(f"{f'[{ci[m][0]:.4f}, {ci[m][1]:.4f}]':>18}" for m in METRIC_NAMES))
    lines.append(f"(ensemble de {result['n_models']} modèle(s), test n = {result['n_test']}, "
                 f"seuils par label appris sur {result['n_oof']} prédictions OOF)")
    return "\n".join(lines)
