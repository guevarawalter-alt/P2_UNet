# -*- coding: utf-8 -*-
"""
notebooks/P2_UNet_STARE.py
==========================
Script principal — Pregunta 2 Examen Parcial RNAPD 2026-I
Segmentación de Vasos Retinianos con U-Net — Dataset STARE

Docente: Ph.D. Aldo Camargo | Fecha: 29 de Mayo 2026

CÓMO EJECUTAR:
  python notebooks/P2_UNet_STARE.py

  En Windows, para ver las barras de progreso correctamente:
    chcp 65001
    python notebooks/P2_UNet_STARE.py
"""

# ════════════════════════════════════════════════════════════════════════════
# CELDA 1 — Configurar entorno y rutas
# ════════════════════════════════════════════════════════════════════════════
import sys, os, subprocess
from pathlib import Path

# La raíz del proyecto es el padre de la carpeta notebooks/
REPO_DIR = Path(__file__).resolve().parent.parent

# Añadir raíz al path para importar src/
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

print(f"Raíz del proyecto: {REPO_DIR}")

# Importar módulos del proyecto
from src import (
    IMG_DIR, MASK_AH_DIR, MASK_VK_DIR, RESULTS_DIR, CHECKPOINTS_DIR,
    IMG_H, IMG_W, SEED, BATCH_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY,
    DROPOUT_P, EARLY_STOP_PAT, SCHED_PATIENCE, SCHED_FACTOR,
    ABLATION_EPOCHS, IMAGENET_MEAN, IMAGENET_STD,
    get_checkpoint_path, get_result_path, verify_data_dirs,
    STAREDataset, load_image, load_mask, apply_clahe,
    denormalize, get_train_transform, get_val_transform,
    UNet, count_params,
    DiceLoss, BCELoss, ComboLoss,
    compute_metrics, print_metrics,
    train_one_epoch, validate, evaluate_full, evaluate_tta,
)
print("Módulos importados correctamente desde src/")

# ════════════════════════════════════════════════════════════════════════════
# CELDA 2 — Verificar datos
# ════════════════════════════════════════════════════════════════════════════
print("\nVerificando datos del repositorio:")
print(f"  Imágenes    : {IMG_DIR}")
print(f"  Máscaras AH : {MASK_AH_DIR}")
print(f"  Máscaras VK : {MASK_VK_DIR}")
print(f"  Resultados  : {RESULTS_DIR}")
print(f"  Checkpoints : {CHECKPOINTS_DIR}")
print()

img_paths, mask_ah, mask_vk = verify_data_dirs()

N_IMAGES = len(img_paths)
print(f"\nPares con máscara (usados para train/val):")
for i, (im, ah, vk) in enumerate(zip(img_paths, mask_ah, mask_vk)):
    print(f"  [{i+1:02d}] {im.name:20s}  AH={ah.name:20s}  VK={vk.name}")
print("\nDataset verificado correctamente")

# Detectar imágenes sin máscara (demo visual)
all_imgs_dir  = sorted(IMG_DIR.glob("*.png"))
labeled_names = {p.stem for p in img_paths}
img_nolabel   = sorted([p for p in all_imgs_dir
                         if p.stem not in labeled_names])
if img_nolabel:
    print(f"\nImágenes sin máscara (demo visual solamente):")
    for p in img_nolabel:
        print(f"  {p.name}")

# ════════════════════════════════════════════════════════════════════════════
# CELDA 3 — Imports y configuración global
# ════════════════════════════════════════════════════════════════════════════
import math, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auc_fn
from scipy.ndimage import binary_dilation, distance_transform_edt as dt_edt
from skimage.morphology import skeletonize
warnings.filterwarnings("ignore")

# Reproducibilidad
np.random.seed(SEED)
torch.manual_seed(SEED)

# Ruta del checkpoint
SAVE_PATH = str(get_checkpoint_path())

# Dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo : {DEVICE}")
if DEVICE.type == "cuda":
    torch.cuda.manual_seed(SEED)
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"VRAM        : "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("Sin GPU — el entrenamiento usará CPU (será más lento)")
print(f"Checkpoint  : {SAVE_PATH}")

# ════════════════════════════════════════════════════════════════════════════
# CELDA 4 — Concordancia inter-anotador y datasets
# ════════════════════════════════════════════════════════════════════════════
print("Calculando concordancia inter-anotador AH vs VK...")
kappas, f1_interanot = [], []

for pa, pv in zip(mask_ah, mask_vk):
    ma = load_mask(pa).ravel()
    mv = load_mask(pv).ravel()
    tp = int(((ma==1)&(mv==1)).sum())
    tn = int(((ma==0)&(mv==0)).sum())
    fp = int(((ma==0)&(mv==1)).sum())
    fn = int(((ma==1)&(mv==0)).sum())
    po  = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    p_e = ((tp+fn)/len(ma))*((tp+fp)/len(ma)) + \
          ((tn+fp)/len(ma))*((tn+fn)/len(ma))
    kappas.append((po - p_e) / (1 - p_e + 1e-8))
    f1_interanot.append(2*tp / (2*tp + fp + fn + 1e-8))

print(f"  Kappa de Cohen medio : {np.mean(kappas):.4f} "
      f"[{min(kappas):.3f}, {max(kappas):.3f}]")
print(f"  F1 inter-anotador    : {np.mean(f1_interanot):.4f} "
      f"[{min(f1_interanot):.3f}, {max(f1_interanot):.3f}]")
kappa_m = np.mean(kappas)
nivel = ("casi perfecto (0.81-1.00)" if kappa_m >= 0.81 else
         "sustancial (0.61-0.80)"    if kappa_m >= 0.61 else "moderado")
print(f"  Acuerdo inter-humano : {nivel}")

vessel_pcts    = [load_mask(m).mean() * 100 for m in mask_ah]
mean_vessel    = np.mean(vessel_pcts)
pos_weight_val = (100 - mean_vessel) / mean_vessel
print(f"\n  Vasos: {mean_vessel:.2f}%  Fondo: {100-mean_vessel:.2f}%")
print(f"  pos_weight BCE: {pos_weight_val:.2f}")

# num_workers=0 en Windows para evitar errores de multiprocessing
NUM_WORKERS = 0 if os.name == "nt" else 2

train_ds = STAREDataset(img_paths, mask_ah,
                         transform=get_train_transform(), use_clahe=True)
val_ds   = STAREDataset(img_paths, mask_vk,
                         transform=get_val_transform(),   use_clahe=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                           shuffle=True,  num_workers=NUM_WORKERS,
                           pin_memory=(DEVICE.type == "cuda"))
val_loader   = DataLoader(val_ds,   batch_size=1,
                           shuffle=False, num_workers=NUM_WORKERS,
                           pin_memory=(DEVICE.type == "cuda"))

print(f"\nTrain: {len(train_ds)} imágenes × máscaras AH")
print(f"Val  : {len(val_ds)}   imágenes × máscaras VK")

# Vista previa
N_PREV = min(3, N_IMAGES)
fig, axes = plt.subplots(N_PREV, 4, figsize=(18, N_PREV * 4))
if N_PREV == 1: axes = axes[np.newaxis]
for row in range(N_PREV):
    img  = load_image(img_paths[row])
    m_ah = load_mask(mask_ah[row])
    m_vk = load_mask(mask_vk[row])
    diff = np.zeros((*m_ah.shape, 3), dtype=np.uint8)
    diff[(m_ah==1)&(m_vk==1)] = [0, 200, 0]
    diff[(m_ah==1)&(m_vk==0)] = [200, 0, 0]
    diff[(m_ah==0)&(m_vk==1)] = [0, 0, 200]
    axes[row,0].imshow(img);               axes[row,0].set_title(img_paths[row].name)
    axes[row,1].imshow(m_ah, cmap="gray"); axes[row,1].set_title("Máscara AH — train")
    axes[row,2].imshow(m_vk, cmap="gray"); axes[row,2].set_title("Máscara VK — val")
    axes[row,3].imshow(diff);              axes[row,3].set_title("Diferencia AH vs VK")
    for ax in axes[row]: ax.axis("off")
leg = [mpatches.Patch(facecolor="green", label="Acuerdo"),
       mpatches.Patch(facecolor="red",   label="Solo AH"),
       mpatches.Patch(facecolor="blue",  label="Solo VK")]
fig.legend(handles=leg, loc="lower center", ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, -0.02))
plt.suptitle("STARE — Imágenes y diferencia AH vs VK", fontweight="bold")
plt.tight_layout()
plt.savefig(str(get_result_path("comparacion_anotadores.png")),
            dpi=150, bbox_inches="tight")
plt.show()
print(f"Figura guardada en: {get_result_path('comparacion_anotadores.png')}")

# ════════════════════════════════════════════════════════════════════════════
# CELDA 5 — Verificar arquitectura U-Net
# ════════════════════════════════════════════════════════════════════════════
_t = UNet(skip_mode="concat", dropout_p=DROPOUT_P).to(DEVICE)
_d = torch.randn(2, 3, IMG_H, IMG_W).to(DEVICE)
print(f"Input  : {_d.shape}")
print(f"Output : {_t(_d).shape}   <- (B, 1, H, W) logits")
print(f"Params : {count_params(_t):,}")
del _t, _d

# ════════════════════════════════════════════════════════════════════════════
# CELDA 6 — Entrenamiento principal
# ════════════════════════════════════════════════════════════════════════════
model     = UNet(in_channels=3, num_classes=1,
                 features=(64, 128, 256, 512),
                 skip_mode="concat", dropout_p=DROPOUT_P).to(DEVICE)
criterion = ComboLoss(alpha=0.5, pos_weight=pos_weight_val,
                      device=str(DEVICE))
optimizer = torch.optim.AdamW(model.parameters(),
                               lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=SCHED_FACTOR, patience=SCHED_PATIENCE)
scaler    = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

history   = dict(train_loss=[], val_loss=[],
                 val_f1=[], val_sens=[], val_spec=[], val_auc=[])
best_f1, patience_ctr = 0.0, 0

print(f"Modelo     : U-Net concat | ComboLoss | AdamW lr={LR}")
print(f"Train/Val  : {len(train_ds)} imgs AH / {len(val_ds)} imgs VK")
print(f"Parámetros : {count_params(model):,}")
print(f"Épocas máx : {NUM_EPOCHS} | Early stop: {EARLY_STOP_PAT}\n")

for epoch in range(1, NUM_EPOCHS + 1):
    lr_now = optimizer.param_groups[0]["lr"]
    print(f"\nÉpoca {epoch:>3}/{NUM_EPOCHS}  lr={lr_now:.2e}")

    tr_loss       = train_one_epoch(model, train_loader, optimizer,
                                     criterion, DEVICE, scaler)
    vl_loss, vl_m = validate(model, val_loader, criterion, DEVICE)
    scheduler.step(vl_m["F1"])

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["val_f1"].append(vl_m["F1"])
    history["val_sens"].append(vl_m["sensitivity"])
    history["val_spec"].append(vl_m["specificity"])
    history["val_auc"].append(vl_m["AUC_ROC"])

    tag = ""
    if vl_m["F1"] > best_f1:
        best_f1 = vl_m["F1"]; patience_ctr = 0
        torch.save(dict(epoch=epoch, model_state=model.state_dict(),
                        optim_state=optimizer.state_dict(),
                        best_f1=best_f1, metrics=vl_m), SAVE_PATH)
        tag = "  MEJOR"
    else:
        patience_ctr += 1

    print(f"  Train={tr_loss:.4f}  Val={vl_loss:.4f}  "
          f"F1={vl_m['F1']:.4f}  Sens={vl_m['sensitivity']:.4f}  "
          f"Spec={vl_m['specificity']:.4f}{tag}")

    if patience_ctr >= EARLY_STOP_PAT:
        print(f"\nEarly stopping en época {epoch}  (mejor F1={best_f1:.4f})")
        break

print(f"\nEntrenamiento completo.  Mejor F1 vs VK = {best_f1:.4f}")

ep_range = range(1, len(history["train_loss"]) + 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(ep_range, history["train_loss"], "b-o", ms=3, label="Train (AH)")
axes[0].plot(ep_range, history["val_loss"],   "r-o", ms=3, label="Val (VK)")
axes[0].set(title="Pérdida ComboLoss", xlabel="Época", ylabel="Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(ep_range, history["val_f1"],   "g-o", ms=3, label="F1")
axes[1].plot(ep_range, history["val_sens"], "m-^", ms=3, label="Sensibilidad")
axes[1].plot(ep_range, history["val_spec"], "c-s", ms=3, label="Especificidad")
axes[1].set(title="Métricas vs VK", xlabel="Época", ylabel="Score")
axes[1].legend(); axes[1].grid(alpha=0.3)
axes[2].plot(ep_range, history["val_auc"], color="darkorange", marker="o", ms=3)
axes[2].set(title="AUC-ROC vs VK", xlabel="Época"); axes[2].grid(alpha=0.3)
plt.suptitle("Curvas de entrenamiento — U-Net STARE (AH a VK)", fontweight="bold")
plt.tight_layout()
plt.savefig(str(get_result_path("training_curves.png")),
            dpi=150, bbox_inches="tight")
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# CELDA 7 — Estudio de Ablación
# ════════════════════════════════════════════════════════════════════════════
ablation_configs = [
    dict(name="Concat + BCE",   skip="concat", loss="bce"),
    dict(name="Concat + Dice",  skip="concat", loss="dice"),
    dict(name="Concat + Combo", skip="concat", loss="combo"),
    dict(name="Sum   + Combo",  skip="sum",    loss="combo"),
]
ablation_results = []

for cfg in ablation_configs:
    print(f"\n{'='*52}\n  {cfg['name']}\n{'='*52}")
    abl_model = UNet(skip_mode=cfg["skip"], dropout_p=DROPOUT_P).to(DEVICE)
    if   cfg["loss"] == "bce":
        abl_crit = BCELoss(pos_weight=pos_weight_val, device=str(DEVICE))
    elif cfg["loss"] == "dice":
        abl_crit = DiceLoss()
    else:
        abl_crit = ComboLoss(0.5, pos_weight_val, device=str(DEVICE))
    abl_opt    = torch.optim.AdamW(abl_model.parameters(),
                                    lr=LR, weight_decay=WEIGHT_DECAY)
    abl_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        abl_opt, T_max=ABLATION_EPOCHS, eta_min=1e-6)
    abl_scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    best_f1_abl, best_m_abl = 0.0, {}
    for ep in range(1, ABLATION_EPOCHS + 1):
        train_one_epoch(abl_model, train_loader, abl_opt,
                        abl_crit, DEVICE, abl_scaler)
        _, vl_m = validate(abl_model, val_loader, abl_crit, DEVICE)
        abl_sched.step()
        if vl_m["F1"] > best_f1_abl:
            best_f1_abl = vl_m["F1"]; best_m_abl = vl_m.copy()
        if ep % 5 == 0:
            print(f"  Ep {ep:>2}: F1={vl_m['F1']:.4f}  "
                  f"Sens={vl_m['sensitivity']:.4f}  "
                  f"Spec={vl_m['specificity']:.4f}")
    ablation_results.append(dict(
        config=cfg["name"],
        F1  =best_m_abl.get("F1", 0),
        Sens=best_m_abl.get("sensitivity", 0),
        Spec=best_m_abl.get("specificity", 0),
        AUC =best_m_abl.get("AUC_ROC", 0)))
    print(f"  Mejor F1 (vs VK): {best_f1_abl:.4f}")
    del abl_model
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

base_f1 = next(r["F1"] for r in ablation_results
               if r["config"] == "Concat + Combo")
print(f"\n{'='*65}")
print(f"{'Config':<22} {'F1':>7} {'Sens':>7} {'Spec':>7} {'AUC':>8} {'dF1':>8}")
print(f"{'-'*65}")
for r in ablation_results:
    delta = (f"{r['F1']-base_f1:+.4f}"
             if r["config"] != "Concat + Combo" else "baseline")
    mark  = " <-" if r["config"] == "Concat + Combo" else ""
    print(f"{r['config']:<22} {r['F1']:>7.4f} {r['Sens']:>7.4f} "
          f"{r['Spec']:>7.4f} {r['AUC']:>8.4f} {delta:>8}{mark}")
print(f"{'='*65}")

x = np.arange(len(ablation_results))
f1_ab  = [r["F1"]  for r in ablation_results]
auc_ab = [r["AUC"] for r in ablation_results]
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - 0.2, f1_ab,  0.35, label="F1-Score",  color="steelblue",  alpha=0.85)
ax.bar(x + 0.2, auc_ab, 0.35, label="AUC-ROC",   color="darkorange", alpha=0.85)
for i, (f1, auc) in enumerate(zip(f1_ab, auc_ab)):
    ax.text(i-0.2, f1 +0.003, f"{f1:.4f}",  ha="center", fontsize=9)
    ax.text(i+0.2, auc+0.003, f"{auc:.4f}", ha="center", fontsize=9)
ax.set(title="Ablación — pérdida x skip (val = VK)",
       ylabel="Score", ylim=[0.4, 1.02])
ax.set_xticks(x)
ax.set_xticklabels([r["config"] for r in ablation_results], fontsize=9)
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(str(get_result_path("ablation_study.png")),
            dpi=150, bbox_inches="tight")
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# CELDA 8 — Evaluación final
# ════════════════════════════════════════════════════════════════════════════
ckpt = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Checkpoint: época {ckpt['epoch']}  F1={ckpt['best_f1']:.4f}")

probs_all, targets_all = evaluate_full(model, val_loader, DEVICE)
metrics_vk = compute_metrics(probs_all, targets_all)
print_metrics(metrics_vk, "EVALUACIÓN FINAL — modelo AH vs máscaras VK")

f1_humano = np.mean(f1_interanot)
print(f"\n  Referencia humana (AH vs VK): {f1_humano:.4f}")
print(f"  Modelo        (AH vs VK)    : {metrics_vk['F1']:.4f}")
print(f"  Brecha modelo vs humano     : {f1_humano - metrics_vk['F1']:.4f}")

rng   = np.random.RandomState(42)
idx_s = rng.choice(len(probs_all), min(500_000, len(probs_all)), replace=False)
fpr, tpr, _ = roc_curve(np.array(targets_all)[idx_s],
                          np.array(probs_all)[idx_s])
roc_auc = auc_fn(fpr, tpr)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(fpr, tpr, "b-", lw=2, label=f"Modelo (AUC={roc_auc:.4f})")
axes[0].fill_between(fpr, tpr, alpha=0.12, color="blue")
axes[0].plot([0,1],[0,1], "k--", lw=1, label="Aleatorio")
axes[0].set(title="Curva ROC — AH entrenado vs VK",
            xlabel="1 - Especificidad", ylabel="Sensibilidad")
axes[0].legend(); axes[0].grid(alpha=0.3)
f1_comp = [metrics_vk["F1"], f1_humano]
axes[1].bar(["Modelo\n(vs VK)", "Inter-\nanotador"], f1_comp,
            color=["steelblue", "seagreen"], alpha=0.85, edgecolor="black", width=0.4)
for i, v in enumerate(f1_comp):
    axes[1].text(i, v+0.005, f"{v:.4f}", ha="center",
                 fontsize=12, fontweight="bold")
axes[1].set(title="F1: modelo vs inter-humano",
            ylabel="F1-Score", ylim=[0.4, 1.0])
axes[1].grid(axis="y", alpha=0.3)
plt.suptitle("Evaluación U-Net STARE", fontweight="bold")
plt.tight_layout()
plt.savefig(str(get_result_path("evaluation_stare.png")),
            dpi=150, bbox_inches="tight")
plt.show()

N_VIS = min(4, N_IMAGES)
fig, axes = plt.subplots(N_VIS, 5, figsize=(22, N_VIS * 4))
if N_VIS == 1: axes = axes[np.newaxis]
with torch.no_grad():
    for row in range(N_VIS):
        img_t,  msk_vk_t = val_ds[row]
        _, msk_ah_t      = train_ds[row]
        logit   = model(img_t.unsqueeze(0).to(DEVICE))
        prob    = torch.sigmoid(logit).squeeze().cpu().numpy()
        pred    = (prob >= 0.5).astype(np.uint8)
        gt_vk   = msk_vk_t.numpy().astype(np.uint8)
        gt_ah   = msk_ah_t.numpy().astype(np.uint8)
        overlay = np.zeros((*pred.shape, 3), dtype=np.uint8)
        overlay[(pred==1)&(gt_vk==1)] = [0, 200, 0]
        overlay[(pred==1)&(gt_vk==0)] = [200, 0, 0]
        overlay[(pred==0)&(gt_vk==1)] = [0, 0, 200]
        axes[row,0].imshow(denormalize(img_t)); axes[row,0].set_title(f"Fundus #{row+1}")
        axes[row,1].imshow(gt_ah, cmap="gray"); axes[row,1].set_title("Máscara AH")
        axes[row,2].imshow(gt_vk, cmap="gray"); axes[row,2].set_title("Máscara VK")
        axes[row,3].imshow(prob, cmap="hot", vmin=0, vmax=1); axes[row,3].set_title("P(vaso)")
        axes[row,4].imshow(overlay);            axes[row,4].set_title("TP=verde FP=rojo FN=azul")
        for ax in axes[row]: ax.axis("off")
leg = [mpatches.Patch(facecolor="green", label="TP"),
       mpatches.Patch(facecolor="red",   label="FP"),
       mpatches.Patch(facecolor="blue",  label="FN")]
fig.legend(handles=leg, loc="lower center", ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, -0.02))
plt.suptitle("Predicciones — AH entrenado, evaluado vs VK", fontweight="bold")
plt.tight_layout()
plt.savefig(str(get_result_path("predictions_stare.png")),
            dpi=150, bbox_inches="tight")
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# CELDA 9 — Experimento de dominio
# ════════════════════════════════════════════════════════════════════════════
val_ds_ah  = STAREDataset(img_paths, mask_ah,
                           transform=get_val_transform(), use_clahe=True)
val_ldr_ah = DataLoader(val_ds_ah, batch_size=1,
                         shuffle=False, num_workers=NUM_WORKERS)
val_ds_nc  = STAREDataset(img_paths, mask_vk,
                           transform=get_val_transform(), use_clahe=False)
val_ldr_nc = DataLoader(val_ds_nc, batch_size=1,
                         shuffle=False, num_workers=NUM_WORKERS)

p1, t1 = evaluate_full(model, val_ldr_ah, DEVICE)
m_ref  = compute_metrics(p1, t1)
print_metrics(m_ref,  "Referencia: AH vs AH")

p2, t2 = evaluate_full(model, val_ldr_nc, DEVICE)
m_nc   = compute_metrics(p2, t2)
print_metrics(m_nc,   "AH vs VK — SIN CLAHE")

p3, t3 = evaluate_full(model, val_loader, DEVICE)
m_cl   = compute_metrics(p3, t3)
print_metrics(m_cl,   "AH vs VK — CON CLAHE")

p4, t4 = evaluate_tta(model, val_ds_nc, DEVICE)
m_tta  = compute_metrics(p4, t4)
print_metrics(m_tta,  "AH vs VK — SIN CLAHE + TTA")

p5, t5 = evaluate_tta(model, val_ds, DEVICE)
m_comb = compute_metrics(p5, t5)
print_metrics(m_comb, "AH vs VK — CLAHE + TTA")

ref_f1   = m_ref["F1"]
gap_base = abs(m_nc["F1"]   - ref_f1)
gap_mit  = abs(m_comb["F1"] - ref_f1)
recov    = (gap_base - gap_mit) / gap_base * 100 if gap_base > 1e-6 else 0
print(f"\nBrecha base (sin CLAHE): {m_nc['F1']-ref_f1:+.4f}")
print(f"Brecha CLAHE+TTA       : {m_comb['F1']-ref_f1:+.4f}")
print(f"Recuperación           : {recov:.1f}%")

configs_dom = [
    ("Referencia AH vs AH",  m_ref,  0.0),
    ("AH vs VK sin CLAHE",   m_nc,   m_nc["F1"]  - ref_f1),
    ("AH vs VK + CLAHE",     m_cl,   m_cl["F1"]  - ref_f1),
    ("AH vs VK + TTA",       m_tta,  m_tta["F1"] - ref_f1),
    ("AH vs VK + CLAHE+TTA", m_comb, m_comb["F1"]- ref_f1),
]
labels_dom = ["Ref AH/AH", "Sin CLAHE", "+CLAHE", "+TTA", "+CLAHE+TTA"]
f1_dom     = [m["F1"] for _, m, _ in configs_dom]
cols_dom   = ["steelblue","tomato","goldenrod","mediumpurple","seagreen"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bars = axes[0].bar(labels_dom, f1_dom, color=cols_dom, alpha=0.85, edgecolor="black")
for bar, val in zip(bars, f1_dom):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
                 f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
axes[0].axhline(ref_f1, ls="--", color="steelblue", lw=1.5,
                label=f"Referencia = {ref_f1:.4f}")
axes[0].set(title="F1 por configuración", ylabel="F1-Score", ylim=[0.4, 1.0])
axes[0].legend(fontsize=9); axes[0].grid(axis="y", alpha=0.3)
gaps = [abs(f - ref_f1) for f in f1_dom[1:]]
axes[1].bar(labels_dom[1:], gaps, color=cols_dom[1:], alpha=0.85, edgecolor="black")
for i, v in enumerate(gaps):
    axes[1].text(i, v+0.001, f"{v:.4f}", ha="center",
                 fontsize=10, fontweight="bold")
axes[1].set(title="Brecha respecto a referencia", ylabel="|delta F1|")
axes[1].grid(axis="y", alpha=0.3)
plt.suptitle("Cambio de dominio — STARE", fontweight="bold")
plt.tight_layout()
plt.savefig(str(get_result_path("domain_experiment.png")),
            dpi=150, bbox_inches="tight")
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# CELDA 10 — Análisis de calibre vascular
# ════════════════════════════════════════════════════════════════════════════
def analyze_by_caliber(model, dataset, device, threshold=0.5):
    model.eval()
    results = {"fino": [], "mediano": [], "grueso": []}
    for idx in range(len(dataset)):
        img_t, msk_t = dataset[idx]
        gt = msk_t.numpy().astype(bool)
        with torch.no_grad():
            prob = torch.sigmoid(
                model(img_t.unsqueeze(0).to(device))
            ).squeeze().cpu().numpy()
        pred     = (prob >= threshold).astype(bool)
        dist_map = dt_edt(gt)
        try:    skel = skeletonize(gt)
        except: skel = np.zeros_like(gt, dtype=bool)
        regions = {
            "fino":    binary_dilation(skel & (dist_map < 3.0),
                                        iterations=2) & gt,
            "mediano": binary_dilation(skel & (dist_map >= 3.0) &
                                        (dist_map < 5.5), iterations=4) & gt,
            "grueso":  binary_dilation(skel & (dist_map >= 5.5),
                                        iterations=7) & gt,
        }
        for cal, region in regions.items():
            if region.sum() < 30: continue
            tp = int(( pred & region).sum())
            fn = int((~pred & region).sum())
            results[cal].append(tp / (tp + fn + 1e-8))
    return {cal: dict(recall=np.mean(v) if v else float("nan"), n=len(v))
            for cal, v in results.items()}

print("Analizando calibre vascular (vs máscaras VK)...")
cal_res = analyze_by_caliber(model, val_ds, DEVICE)
cal_labels = {
    "fino":    "Capilares  (< 3 px)",
    "mediano": "Arteriolas (3-5 px)",
    "grueso":  "Arterias   (> 5 px)",
}
print(f"\n{'='*48}\n  Recall por calibre — vs máscaras VK\n{'-'*48}")
for k, label in cal_labels.items():
    r   = cal_res[k]
    bar = chr(9608) * int(r["recall"]*20) if not np.isnan(r["recall"]) else "-"
    print(f"  {label}: {r['recall']:.4f}  {bar}")
print(f"{'='*48}")

colors_cal  = ["#E53935", "#FB8C00", "#43A047"]
recalls_cal = [cal_res[k]["recall"] for k in cal_res]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(list(cal_labels.values()), recalls_cal,
            color=colors_cal, alpha=0.85, edgecolor="black")
for i, v in enumerate(recalls_cal):
    if not np.isnan(v):
        axes[0].text(i, v+0.01, f"{v:.3f}", ha="center",
                     fontsize=11, fontweight="bold")
axes[0].set(title="Recall por calibre vascular",
            ylabel="Recall", ylim=[0.2, 1.05])
axes[0].grid(axis="y", alpha=0.3)
radios = np.linspace(0.5, 10, 200)
axes[1].plot(radios, 0.95*(1-np.exp(-radios/2.8)),
             color="steelblue", lw=2.5, label="Tendencia empirica")
x_pts = [1.5, 4.0, 8.0]
y_pts = [cal_res["fino"]["recall"], cal_res["mediano"]["recall"],
         cal_res["grueso"]["recall"]]
axes[1].scatter(x_pts, y_pts, s=130, c=colors_cal, zorder=5,
                edgecolors="white", lw=1.5)
for xi, yi, lab in zip(x_pts, y_pts, ["Capilares","Arteriolas","Arterias"]):
    if not np.isnan(yi):
        axes[1].annotate(lab+f" R={yi:.3f}", xy=(xi,yi),
                         xytext=(xi+0.5,yi-0.09), fontsize=8.5,
                         arrowprops=dict(arrowstyle="->",color="gray",lw=1))
axes[1].axvline(3.0, ls=":", color="gray", lw=1)
axes[1].axvline(5.5, ls=":", color="gray", lw=1)
axes[1].set(title="Recall vs. radio vascular (px)",
            xlabel="Radio (px)", ylabel="Recall", ylim=[0.2, 1.05])
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.suptitle("Fallos por calibre vascular — STARE (vs VK)", fontweight="bold")
plt.tight_layout()
plt.savefig(str(get_result_path("caliber_analysis.png")),
            dpi=150, bbox_inches="tight")
plt.show()

@torch.no_grad()
def per_image_f1(model, dataset, device):
    model.eval()
    results = []
    for idx in range(len(dataset)):
        img_t, msk_t = dataset[idx]
        prob = torch.sigmoid(
            model(img_t.unsqueeze(0).to(device))
        ).squeeze().cpu().numpy()
        pred = (prob >= 0.5).astype(np.uint8)
        gt   = msk_t.numpy().astype(np.uint8)
        tp = int(((pred==1)&(gt==1)).sum())
        fp = int(((pred==1)&(gt==0)).sum())
        fn = int(((pred==0)&(gt==1)).sum())
        results.append(dict(idx=idx,
                             f1=2*tp/(2*tp+fp+fn+1e-8),
                             recall=tp/(tp+fn+1e-8),
                             fp=fp, fn=fn, prob=prob))
    return sorted(results, key=lambda x: x["f1"])

img_results = per_image_f1(model, val_ds, DEVICE)
N_FAIL      = min(3, N_IMAGES)
print(f"\nTop-{N_FAIL} imágenes con peor F1 (vs VK):")
for r in img_results[:N_FAIL]:
    print(f"  Idx={r['idx']}  F1={r['f1']:.4f}  "
          f"Recall={r['recall']:.4f}  FP={r['fp']:,}  FN={r['fn']:,}")

hypotheses = [
    "Capilar muy fino (<2 px): contraste post-CLAHE insuficiente.",
    "VK anota capilares que AH no marco: FN inevitables.",
    "Disco optico brillante: activaciones falsas del decoder.",
]
fig, axes_f = plt.subplots(N_FAIL, 5, figsize=(22, N_FAIL * 4))
if N_FAIL == 1: axes_f = axes_f[np.newaxis]
for row, (res, hyp) in enumerate(zip(img_results[:N_FAIL], hypotheses)):
    img_t,  msk_vk_t = val_ds[res["idx"]]
    _, msk_ah_t      = train_ds[res["idx"]]
    gt_vk = msk_vk_t.numpy().astype(np.uint8)
    gt_ah = msk_ah_t.numpy().astype(np.uint8)
    prob  = res["prob"]; pred = (prob >= 0.5).astype(np.uint8)
    overlay = np.zeros((*pred.shape, 3), dtype=np.uint8)
    overlay[(pred==1)&(gt_vk==1)] = [0, 200, 0]
    overlay[(pred==1)&(gt_vk==0)] = [200, 0, 0]
    overlay[(pred==0)&(gt_vk==1)] = [0, 0, 200]
    axes_f[row,0].imshow(denormalize(img_t))
    axes_f[row,0].set_title(f"Img #{res['idx']}  F1={res['f1']:.4f}")
    axes_f[row,1].imshow(gt_ah, cmap="gray"); axes_f[row,1].set_title("Mascara AH")
    axes_f[row,2].imshow(gt_vk, cmap="gray"); axes_f[row,2].set_title("Mascara VK")
    axes_f[row,3].imshow(prob, cmap="hot", vmin=0, vmax=1)
    axes_f[row,3].set_title("P(vaso)")
    axes_f[row,4].imshow(overlay); axes_f[row,4].set_title("Mapa de errores")
    for ax in axes_f[row]: ax.axis("off")
    print(f"\nCaso {row+1}: {hyp}")
leg = [mpatches.Patch(facecolor="green", label="TP"),
       mpatches.Patch(facecolor="red",   label="FP"),
       mpatches.Patch(facecolor="blue",  label="FN")]
fig.legend(handles=leg, loc="lower center", ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, -0.02))
plt.suptitle("Casos de fallo — AH entrenado, evaluado vs VK", fontweight="bold")
plt.tight_layout()
plt.savefig(str(get_result_path("failure_analysis.png")),
            dpi=150, bbox_inches="tight")
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# CELDA 11 — Resumen final
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*66)
print("  RESUMEN — Pregunta 2 Examen Parcial 2026-I")
print(f"  STARE | {N_IMAGES} imgs | AH entrena, VK valida")
print("="*66)
print(f"\n  E1: U-Net   — Parametros: {count_params(model):,}  skip: concat")
for r in ablation_results:
    mark = " <- optimo" if r["config"] == "Concat + Combo" else ""
    print(f"  E2:  {r['config']:<22}: F1={r['F1']:.4f}  AUC={r['AUC']:.4f}{mark}")
print(f"\n  E3: Evaluacion final")
print(f"    Sensibilidad : {metrics_vk['sensitivity']:.4f}")
print(f"    Especificidad: {metrics_vk['specificity']:.4f}")
print(f"    F1-Score     : {metrics_vk['F1']:.4f}")
print(f"    AUC-ROC      : {metrics_vk['AUC_ROC']:.4f}")
print(f"    Inter-anot.  : {np.mean(f1_interanot):.4f} "
      f"(Kappa={np.mean(kappas):.4f})")
print(f"\n  E4+E6: Brecha sin CLAHE: {m_nc['F1']-m_ref['F1']:+.4f}  "
      f"CLAHE+TTA: {m_comb['F1']-m_ref['F1']:+.4f}  "
      f"Recuperacion: {recov:.1f}%")
print(f"\n  E5: Calibre vascular")
for k, label in cal_labels.items():
    print(f"    {label}: Recall={cal_res[k]['recall']:.4f}")
print(f"\n  Archivos generados en {RESULTS_DIR}:")
for f in ["comparacion_anotadores.png","training_curves.png",
          "ablation_study.png","evaluation_stare.png",
          "predictions_stare.png","domain_experiment.png",
          "caliber_analysis.png","failure_analysis.png"]:
    p = get_result_path(f)
    print(f"    {'OK' if p.exists() else 'pendiente':8s} {f}")
print("\n" + "="*66)
