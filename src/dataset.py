# -*- coding: utf-8 -*-
"""
src/dataset.py
==============
Clases Dataset y funciones de carga/preprocesamiento para STARE.

Estrategia AH→VK:
  STAREDataset(img_paths, mask_ah) → entrenamiento
  STAREDataset(img_paths, mask_vk) → validación cuantitativa

Pregunta 2 — Examen Parcial RNAPD 2026-I
Docente: Ph.D. Aldo Camargo
"""

import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import (IMG_H, IMG_W, IMAGENET_MEAN, IMAGENET_STD,
                     DROPOUT_P)


# ── Funciones de carga ────────────────────────────────────────────────────
def load_image(path) -> np.ndarray:
    """Carga imagen RGB uint8. Soporta PNG y JPEG."""
    return np.array(Image.open(str(path)).convert("RGB"), dtype=np.uint8)


def load_mask(path) -> np.ndarray:
    """Carga máscara binaria uint8. Umbral en 127."""
    mask = np.array(Image.open(str(path)).convert("L"), dtype=np.uint8)
    return (mask > 127).astype(np.uint8)


def apply_clahe(image_rgb: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Aplica CLAHE al canal verde de una imagen RGB.

    Justificación: la hemoglobina tiene absorbancia máxima ~550 nm
    (canal verde), donde los vasos retinianos tienen mayor contraste.

    Parámetros
    ----------
    clip_limit : amplificación máxima del contraste (2.0 = 200%)
    tile_grid  : tiles de ecualización local (8×8 → tiles de 64×64 px)
    """
    clahe           = cv2.createCLAHE(clipLimit=clip_limit,
                                       tileGridSize=tile_grid)
    result          = image_rgb.copy()
    result[:, :, 1] = clahe.apply(image_rgb[:, :, 1])
    return result


def denormalize(tensor) -> np.ndarray:
    """Desnormaliza un tensor ImageNet y devuelve array HWC float [0,1]."""
    arr = tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(arr * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


# ── Transforms ────────────────────────────────────────────────────────────
def get_train_transform():
    """
    Pipeline de augmentation para entrenamiento.
    Agresivo porque solo hay 10 imágenes disponibles.
    """
    return A.Compose([
        A.Resize(IMG_H, IMG_W),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussNoise(var_limit=(5, 25), p=0.4),
        A.ElasticTransform(alpha=40, sigma=6, p=0.3),
        A.GridDistortion(p=0.2),
        A.OpticalDistortion(p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transform():
    """Pipeline mínimo para validación: solo resize y normalización."""
    return A.Compose([
        A.Resize(IMG_H, IMG_W),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────
class STAREDataset(Dataset):
    """
    Dataset para imágenes de retina STARE con máscara de segmentación.

    Uso típico:
        train_ds = STAREDataset(img_paths, mask_ah,
                                transform=get_train_transform())
        val_ds   = STAREDataset(img_paths, mask_vk,
                                transform=get_val_transform())

    Parámetros
    ----------
    img_paths  : list[Path] — rutas a imágenes .png
    mask_paths : list[Path] — rutas a máscaras .jpg
    transform  : albumentations.Compose — augmentation pipeline
    use_clahe  : bool — aplicar CLAHE al canal verde antes de transformar
    """
    def __init__(self, img_paths, mask_paths,
                 transform=None, use_clahe: bool = True):
        assert len(img_paths) == len(mask_paths), (
            f"Imágenes ({len(img_paths)}) y "
            f"máscaras ({len(mask_paths)}) no coinciden.")
        self.img_paths  = list(img_paths)
        self.mask_paths = list(mask_paths)
        self.transform  = transform
        self.use_clahe  = use_clahe

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        image = load_image(self.img_paths[idx])
        mask  = load_mask(self.mask_paths[idx])

        if self.use_clahe:
            image = apply_clahe(image)

        if self.transform:
            out   = self.transform(image=image, mask=mask)
            image = out["image"]
            mask  = out["mask"].float()
        else:
            image = torch.from_numpy(
                image.transpose(2, 0, 1)).float() / 255.
            mask  = torch.from_numpy(mask).float()

        return image, mask   # (C,H,W) float, (H,W) float {0,1}
