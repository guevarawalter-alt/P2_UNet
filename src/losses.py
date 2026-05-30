# -*- coding: utf-8 -*-
"""
src/losses.py
=============
Funciones de pérdida para segmentación binaria con desbalance de clases.

Implementadas:
  DiceLoss   — optimiza directamente la superposición F1
  BCELoss    — entropía cruzada binaria con pos_weight
  ComboLoss  — 0.5·BCE + 0.5·Dice ← configuración ganadora del estudio

Pregunta 2 — Examen Parcial RNAPD 2026-I
Docente: Ph.D. Aldo Camargo
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss para segmentación binaria.

        L = 1 - (2·TP + smooth) / (2·TP + FP + FN + smooth)

    Opera sobre probabilidades continuas (tras sigmoid) → diferenciable.
    Robusta al desbalance: no depende del conteo de píxeles sino
    de la superposición de segmentos.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1).float()
        inter   = (probs * targets).sum()
        union   = probs.sum() + targets.sum()
        return 1 - (2 * inter + self.smooth) / (union + self.smooth)


class BCELoss(nn.Module):
    """
    Entropía cruzada binaria con ponderación de clase positiva.

        L = -[pos_weight·y·log(p) + (1-y)·log(1-p)]

    Con pos_weight = ratio fondo/vasos ≈ 2.07 en STARE (máscaras AH):
      → gradiente de píxel de vaso pesa 2.07× más que el de fondo
      → compensa el desbalance 32.6% vasos / 67.4% fondo
    """
    def __init__(self, pos_weight: float = None, device: str = "cpu"):
        super().__init__()
        w = (torch.tensor([pos_weight]).to(device)
             if pos_weight is not None else None)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=w)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets.unsqueeze(1).float())


class ComboLoss(nn.Module):
    """
    ComboLoss = alpha·BCE + (1 - alpha)·Dice

    Configuración ganadora del estudio de ablación: alpha=0.5.

    Ventaja de la combinación:
      BCE  → estabilidad en épocas tempranas (gradientes píxel a píxel)
      Dice → precisión en convergencia (superposición global del segmento)

    Referencia:
        Taghanaki et al. (2019). Combo loss: Handling input and output
        imbalance in multi-organ segmentation. CMIG, 75, 24-33.
    """
    def __init__(self, alpha: float = 0.5,
                 pos_weight: float = None,
                 device: str = "cpu"):
        super().__init__()
        self.alpha   = alpha
        w = (torch.tensor([pos_weight]).to(device)
             if pos_weight is not None else None)
        self.bce_fn  = nn.BCEWithLogitsLoss(pos_weight=w)
        self.dice_fn = DiceLoss()

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bce  = self.bce_fn(logits, targets.unsqueeze(1).float())
        dice = self.dice_fn(logits, targets)
        return self.alpha * bce + (1 - self.alpha) * dice
