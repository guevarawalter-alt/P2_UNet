# -*- coding: utf-8 -*-
"""
src/__init__.py
===============
Paquete principal del proyecto P2_UNet.
Pregunta 2 — Examen Parcial RNAPD 2026-I
Docente: Ph.D. Aldo Camargo
"""

from .config  import (REPO_ROOT, DATA_DIR, IMG_DIR, MASK_AH_DIR, MASK_VK_DIR,
                       RESULTS_DIR, CHECKPOINTS_DIR, IMG_H, IMG_W,
                       SEED, BATCH_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY,
                       DROPOUT_P, EARLY_STOP_PAT, SCHED_PATIENCE,
                       SCHED_FACTOR, ABLATION_EPOCHS, IMAGENET_MEAN,
                       IMAGENET_STD, CHECKPOINT_NAME,
                       get_checkpoint_path, get_result_path, verify_data_dirs)

from .dataset import (STAREDataset, load_image, load_mask, apply_clahe,
                       denormalize, get_train_transform, get_val_transform)

from .model   import (UNet, EncoderBlock, DecoderBlockConcat,
                       DecoderBlockSum, double_conv, count_params)

from .losses  import DiceLoss, BCELoss, ComboLoss

from .metrics import compute_metrics, print_metrics

from .train   import (train_one_epoch, validate,
                       evaluate_full, evaluate_tta)
