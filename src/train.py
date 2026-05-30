# -*- coding: utf-8 -*-
"""
src/train.py
============
Funciones de entrenamiento, validación y evaluación (con y sin TTA).

Funciones exportadas:
  train_one_epoch  — una época de entrenamiento con AMP + GradScaler
  validate         — evaluación con métricas completas
  evaluate_full    — solo probabilidades (sin calcular pérdida)
  evaluate_tta     — Test-Time Augmentation: promedio de 8 transformaciones

Pregunta 2 — Examen Parcial RNAPD 2026-I
Docente: Ph.D. Aldo Camargo
"""

import sys
import numpy as np
import torch
from tqdm import tqdm

from .metrics import compute_metrics


def train_one_epoch(model, loader, optimizer, criterion,
                    device, scaler) -> float:
    """
    Ejecuta una época de entrenamiento con Mixed Precision (AMP).

    AMP usa float16 en operaciones de convolución (más rápido)
    y float32 donde la precisión es crítica (pérdida, actualización de pesos).
    GradScaler previene el underflow de gradientes en float16.

    Retorna
    -------
    float : pérdida media de la época
    """
    model.train()
    total_loss = 0.0

    for images, masks in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device)
        masks  = masks.to(device).float()
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            loss = criterion(model(images), masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device,
             threshold: float = 0.5) -> tuple:
    """
    Evalúa el modelo sobre el conjunto de validación.

    Retorna
    -------
    tuple : (pérdida_media, dict_de_métricas)
    """
    model.eval()
    total_loss  = 0.0
    all_probs   = []
    all_targets = []

    for images, masks in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device)
        masks  = masks.to(device).float()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            total_loss += criterion(logits, masks).item()

        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_probs.extend(probs.ravel().tolist())
        all_targets.extend(masks.cpu().numpy().ravel().tolist())

    metrics = compute_metrics(all_probs, all_targets, threshold)
    return total_loss / len(loader), metrics


@torch.no_grad()
def evaluate_full(model, loader, device,
                  threshold: float = 0.5) -> tuple:
    """
    Evaluación completa sin calcular función de pérdida.
    Útil para el análisis de dominio y calibre vascular.

    Retorna
    -------
    tuple : (all_probs, all_targets) como listas de floats
    """
    model.eval()
    all_probs   = []
    all_targets = []

    for images, masks in tqdm(loader, desc="Evaluando", leave=False):
        images = images.to(device)
        logits = model(images)
        probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_probs.extend(probs.ravel().tolist())
        all_targets.extend(masks.numpy().ravel().tolist())

    return all_probs, all_targets


@torch.no_grad()
def evaluate_tta(model, dataset, device,
                 threshold: float = 0.5) -> tuple:
    """
    Evaluación con Test-Time Augmentation (TTA).

    Aplica 8 transformaciones geométricas a cada imagen:
        4 rotaciones (0°, 90°, 180°, 270°) × 2 estados de flip horizontal
    Promedia las 8 predicciones para reducir la varianza de predicción.

    Efecto en STARE:
      Especificidad ↑ ~3.8 pp — elimina FP inestables por promediado
      Sensibilidad  ↓ ~5.3 pp — atenúa capilares de baja confianza

    Parámetros
    ----------
    model   : UNet entrenado
    dataset : STAREDataset (acceso por índice, no DataLoader)
    device  : torch.device

    Retorna
    -------
    tuple : (all_probs, all_targets)
    """
    model.eval()
    all_probs   = []
    all_targets = []

    for idx in tqdm(range(len(dataset)), desc="TTA", leave=False):
        img_t, msk_t = dataset[idx]
        img_b = img_t.unsqueeze(0).to(device)

        preds_tta = []
        for k in range(4):
            rot = torch.rot90(img_b, k=k, dims=[2, 3])
            for flip in [False, True]:
                aug  = torch.flip(rot, dims=[3]) if flip else rot
                prob = torch.sigmoid(model(aug)).squeeze().cpu().numpy()
                if flip:
                    prob = np.flip(prob, axis=1)
                prob = np.rot90(prob, k=-k).copy()
                preds_tta.append(prob)

        avg_prob = np.mean(preds_tta, axis=0)
        all_probs.extend(avg_prob.ravel().tolist())
        all_targets.extend(msk_t.numpy().ravel().tolist())

    return all_probs, all_targets
