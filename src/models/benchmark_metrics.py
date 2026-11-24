"""Évaluation de Han et al. 2026 (arXiv:2608.25893) : un seuil par label appris sur les prédictions
out-of-fold, métriques macro sur le test, intervalle de confiance à 95 % par 1000 tirages bootstrap."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def label_thresholds(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Seuil qui maximise le F1 de chaque label."""
    thresholds = np.full(targets.shape[1], 0.5)
    for j in range(targets.shape[1]):
        if targets[:, j].any():
            p, r, t = precision_recall_curve(targets[:, j], probs[:, j])
            thresholds[j] = t[np.argmax(2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12))]
    return thresholds


def macro_metrics(probs: np.ndarray, targets: np.ndarray, thresholds: np.ndarray) -> dict[str, float]:
    defined = [j for j in range(targets.shape[1]) if 0 < targets[:, j].sum() < len(targets)]  # AUC calculable
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, probs >= thresholds, average="macro", zero_division=0
    )
    return {
        "auroc": np.mean([roc_auc_score(targets[:, j], probs[:, j]) for j in defined]),
        "auprc": np.mean([average_precision_score(targets[:, j], probs[:, j]) for j in defined]),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def evaluate(oof_probs, oof_targets, test_probs, test_targets, n_boot: int = 1000, seed: int = 0) -> dict:
    """Retourne {métrique: [valeur, borne basse, borne haute]} sur le test."""
    thresholds = label_thresholds(oof_probs, oof_targets)
    rng = np.random.default_rng(seed)
    boots = [
        macro_metrics(test_probs[i], test_targets[i], thresholds)
        for i in rng.integers(0, len(test_targets), (n_boot, len(test_targets)))
    ]
    scores = macro_metrics(test_probs, test_targets, thresholds)
    return {k: [float(v), *np.percentile([b[k] for b in boots], [2.5, 97.5]).tolist()] for k, v in scores.items()}
