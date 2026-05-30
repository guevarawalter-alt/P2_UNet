# -*- coding: utf-8 -*-
"""
src/metrics.py
==============
Métricas de evaluación para segmentación binaria de vasos retinianos.

Pregunta 2 — Examen Parcial RNAPD 2026-I
Docente: Ph.D. Aldo Camargo
"""

import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def compute_metrics(all_probs: list,
                    all_targets: list,
                    threshold: float = 0.5) -> dict:
    """
    Calcula métricas de segmentación a partir de probabilidades predichas.

    Parámetros
    ----------
    all_probs   : probabilidades P(vaso) por píxel, en [0, 1]
    all_targets : etiquetas reales (0=fondo, 1=vaso)
    threshold   : umbral de binarización

    Retorna
    -------
    dict con: sensitivity, specificity, precision, F1,
              accuracy, AUC_ROC, TP, TN, FP, FN
    """
    probs   = np.array(all_probs).ravel()
    targets = np.array(all_targets).ravel().astype(int)
    preds   = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()

    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    f1   = 2 * tp / (2 * tp + fp + fn + 1e-8)
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = float("nan")

    return dict(
        sensitivity=sens, specificity=spec, precision=prec,
        F1=f1, accuracy=acc, AUC_ROC=auc,
        TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
    )


def print_metrics(m: dict, title: str = "") -> None:
    """Imprime el diccionario de métricas de forma legible."""
    sep = "─" * 54
    print(f"\n{sep}")
    if title:
        print(f"  {title}")
    print(sep)
    print(f"  Sensibilidad (Recall): {m['sensitivity']:.4f}")
    print(f"  Especificidad        : {m['specificity']:.4f}")
    print(f"  Precisión            : {m['precision']:.4f}")
    print(f"  F1-Score             : {m['F1']:.4f}")
    print(f"  Accuracy             : {m['accuracy']:.4f}")
    print(f"  AUC-ROC              : {m['AUC_ROC']:.4f}")
    print(f"  TP={m['TP']:,}  TN={m['TN']:,}  "
          f"FP={m['FP']:,}  FN={m['FN']:,}")
    print(sep)
