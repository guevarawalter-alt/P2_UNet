# -*- coding: utf-8 -*-
"""
src/config.py
=============
Configuración centralizada del proyecto.
Todas las rutas apuntan a carpetas internas del repositorio.

Estructura esperada del repositorio:
    P2_UNet/
        data/
            images/       ← imágenes .png  (im0001..im0139)
            labels_ah/    ← máscaras .jpg  experto Hoover
            labels_vk/    ← máscaras .jpg  experto Kouznetsova
        results/          ← figuras generadas
        checkpoints/      ← pesos del modelo (.pth)
        src/              ← código fuente
        notebooks/        ← notebook principal
"""

from pathlib import Path

# ── Raíz del repositorio ──────────────────────────────────────────────────
# Sube un nivel desde src/ para llegar a la raíz del repo
REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Rutas de datos (internas al repositorio) ──────────────────────────────
DATA_DIR    = REPO_ROOT / "data"
IMG_DIR     = DATA_DIR  / "images"       # imágenes .png
MASK_AH_DIR = DATA_DIR  / "labels_ah"   # máscaras experto AH (.jpg)
MASK_VK_DIR = DATA_DIR  / "labels_vk"   # máscaras experto VK (.jpg)

# ── Rutas de salida (internas al repositorio) ─────────────────────────────
RESULTS_DIR     = REPO_ROOT / "results"      # figuras generadas
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"  # pesos del modelo

# ── Hiperparámetros de imagen ─────────────────────────────────────────────
IMG_H = 512
IMG_W = 512

# ── Hiperparámetros de entrenamiento ──────────────────────────────────────
SEED            = 42
BATCH_SIZE      = 2
NUM_EPOCHS      = 6
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
DROPOUT_P       = 0.2
EARLY_STOP_PAT  = 15   # épocas sin mejora antes de parar
SCHED_PATIENCE  = 7    # épocas sin mejora antes de reducir lr
SCHED_FACTOR    = 0.5  # factor de reducción del lr
ABLATION_EPOCHS = 2   # épocas por configuración de ablación

# ── Normalización ImageNet ────────────────────────────────────────────────
import numpy as np
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Nombre del checkpoint ─────────────────────────────────────────────────
CHECKPOINT_NAME = "unet_stare_best.pth"


def get_checkpoint_path() -> Path:
    """Devuelve la ruta completa al checkpoint del mejor modelo."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINTS_DIR / CHECKPOINT_NAME


def get_result_path(filename: str) -> Path:
    """Devuelve la ruta completa a un archivo de resultados."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / filename


def verify_data_dirs() -> tuple:
    """
    Verifica que las carpetas de datos existen y contienen archivos.

    Estrategia STARE (20 imgs totales, 10 con máscara de cada experto):
      - data/images/    puede tener 10 o 20 imágenes .png
      - data/labels_ah/ tiene 10 máscaras .jpg del experto AH
      - data/labels_vk/ tiene 10 máscaras .jpg del experto VK

    Las imágenes sin máscara (si hay 20 en total) se detectan
    automáticamente comparando nombres de archivo y se separan
    para uso en demostración visual (sin evaluación cuantitativa).

    Retorna
    -------
    tuple : (img_labeled, mask_ah, mask_vk)
        Solo las imágenes que tienen máscara en AMBOS expertos.
    """
    all_imgs = sorted(IMG_DIR.glob("*.png"))
    mask_ah  = sorted(MASK_AH_DIR.glob("*.jpg"))
    mask_vk  = sorted(MASK_VK_DIR.glob("*.jpg"))

    # Validaciones básicas
    assert len(all_imgs) > 0, (
        f"\nNo se encontraron imágenes .png en:\n  {IMG_DIR}\n"
        "Sube las imágenes STARE a data/images/ en el repositorio.")
    assert len(mask_ah) > 0, (
        f"\nNo se encontraron máscaras .jpg en:\n  {MASK_AH_DIR}\n"
        "Sube las máscaras AH a data/labels_ah/ en el repositorio.")
    assert len(mask_vk) > 0, (
        f"\nNo se encontraron máscaras .jpg en:\n  {MASK_VK_DIR}\n"
        "Sube las máscaras VK a data/labels_vk/ en el repositorio.")
    assert len(mask_ah) == len(mask_vk), (
        f"Máscaras AH ({len(mask_ah)}) y VK ({len(mask_vk)}) "
        f"no tienen el mismo número de archivos.")

    # Identificar qué imágenes tienen máscara en ambos expertos
    # comparando los stems (nombres sin extensión)
    names_ah  = {p.stem for p in mask_ah}
    names_vk  = {p.stem for p in mask_vk}
    labeled   = names_ah & names_vk          # intersección: tienen ambas

    # Imágenes con máscara (para entrenamiento y validación cuantitativa)
    img_by_name  = {p.stem: p for p in all_imgs}
    img_labeled  = sorted([img_by_name[n] for n in sorted(labeled)
                            if n in img_by_name])
    mask_ah_filt = sorted([p for p in mask_ah if p.stem in labeled])
    mask_vk_filt = sorted([p for p in mask_vk if p.stem in labeled])

    # Imágenes SIN máscara (para demo visual solamente)
    unlabeled    = {p.stem for p in all_imgs} - labeled
    img_unlabeled = sorted([img_by_name[n] for n in sorted(unlabeled)
                             if n in img_by_name])

    # Resumen
    print(f"\n  Imágenes totales en data/images/ : {len(all_imgs)}")
    print(f"  Imágenes CON máscara (train/val) : {len(img_labeled)}")
    print(f"  Imágenes SIN máscara (demo visual): {len(img_unlabeled)}")
    print(f"  Máscaras AH                       : {len(mask_ah_filt)}")
    print(f"  Máscaras VK                       : {len(mask_vk_filt)}")

    assert len(img_labeled) > 0, (
        "No se encontró ninguna imagen cuyo nombre coincida con "
        "las máscaras. Verifica que los nombres de archivo son iguales "
        "en images/, labels_ah/ y labels_vk/ (ej. im0001.png / im0001.jpg).")

    return img_labeled, mask_ah_filt, mask_vk_filt


if __name__ == "__main__":
    print("=== Configuración del Proyecto ===")
    print(f"Raíz del repo : {REPO_ROOT}")
    print(f"Imágenes      : {IMG_DIR}")
    print(f"Máscaras AH   : {MASK_AH_DIR}")
    print(f"Máscaras VK   : {MASK_VK_DIR}")
    print(f"Resultados    : {RESULTS_DIR}")
    print(f"Checkpoints   : {CHECKPOINTS_DIR}")
    print(f"Checkpoint    : {get_checkpoint_path()}")
