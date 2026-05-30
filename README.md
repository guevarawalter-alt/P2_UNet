# P2 — Segmentación de Vasos Retinianos con U-Net

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-orange.svg)](https://pytorch.org)
[![Dataset STARE](https://img.shields.io/badge/dataset-STARE-green.svg)](https://cecas.clemson.edu/~ahoover/stare/)

**Curso:** Redes Neuronales y Aprendizaje Profundo — Examen Parcial 2026-I
**Docente:** Ph.D. Aldo Camargo | **Entrega:** 30 de Mayo del 2026

---

## Ejecutar el proyecto

```bash
# 1. Clonar el repositorio
git clone https://github.com/guevarawalter-alt/P2_UNet.git
cd P2_UNet

# 2. Crear entorno virtual e instalar dependencias
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux
pip install -r requirements.txt

# 3. Ejecutar
python notebooks/P2_UNet_STARE.py
```

> **Windows:** ejecuta `chcp 65001` antes del script para ver las barras de progreso correctamente.

---

## Resultados principales

| Métrica       | Nuestro modelo | Inter-anotador | Ref. U-Net lit. |
|---------------|:--------------:|:--------------:|:---------------:|
| **F1-Score**  | **0.9041**     | 0.9720         | 0.8039          |
| **AUC-ROC**   | **0.9821**     | —              | 0.9790          |
| Sensibilidad  | 0.9299         | ~0.9720        | 0.7821          |
| Especificidad | 0.9383         | ~0.9720        | 0.9806          |
| Kappa (AH/VK) | —              | **0.9660**     | —               |

---

## Estructura del repositorio

```
P2_UNet/
├── src/
│   ├── config.py       ← rutas internas + hiperparámetros
│   ├── dataset.py      ← STAREDataset + CLAHE + transforms
│   ├── model.py        ← U-Net, EncoderBlock, DecoderBlock
│   ├── losses.py       ← DiceLoss, BCELoss, ComboLoss
│   ├── metrics.py      ← compute_metrics, print_metrics
│   ├── train.py        ← train_one_epoch, validate, TTA
│   └── __init__.py
├── notebooks/
│   └── P2_UNet_STARE.py   ← script principal
├── data/
│   ├── images/         ← imágenes .png  [subir manualmente]
│   ├── labels_ah/      ← mascaras .jpg experto AH [subir manualmente]
│   └── labels_vk/      ← mascaras .jpg experto VK [subir manualmente]
├── results/            ← figuras generadas [subir manualmente]
├── checkpoints/        ← pesos del modelo (.pth)
├── requirements.txt
└── .gitignore
```

---

## Cómo funcionan las rutas

Todas las rutas están definidas en `src/config.py` relativas a la raíz del proyecto:

```python
REPO_ROOT   = Path(__file__).resolve().parent.parent
IMG_DIR     = REPO_ROOT / "data" / "images"
MASK_AH_DIR = REPO_ROOT / "data" / "labels_ah"
MASK_VK_DIR = REPO_ROOT / "data" / "labels_vk"
RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
```

Sin rutas absolutas. Funciona en Windows, Mac y Linux sin cambiar nada.

---

## Estrategia AH a VK

```
Dataset STARE — 10 imágenes:
  10 con mascara de ambos expertos → entrenamiento y validación
  10 sin mascara                   → demo visual

  AH (A. Hoover, Clemson Univ.)  → Entrenamiento: 10 mascaras
  VK (V. Kouznetsova, UCSD)      → Validación cuantitativa: 10 mascaras

Kappa de Cohen AH/VK: 0.9660 (acuerdo casi perfecto)
F1 inter-anotador  : 0.9720 (techo teórico del modelo)
```

---

## Entregables

| # | Entregable | Resultado |
|---|-----------|-----------|
| E1 | U-Net desde cero en PyTorch | 31M params, concat optimo |
| E2 | Estudio de ablación (4 configs) | Combo+concat: F1=0.9007 |
| E3 | Evaluación: F1, AUC-ROC, sensibilidad, especificidad | F1=0.9041, AUC=0.9821 |
| E4 | Experimento de dominio AH a VK | 5 condiciones evaluadas |
| E5 | Análisis por calibre vascular | Capilares 0.70 · Arterias 0.94 |
| E6 | Adaptación: CLAHE + TTA | Recuperación 77.7% de la brecha |

---

## Dataset STARE

Hoover, A., Kouznetsova, V., & Goldbaum, M. (2000).
*IEEE Transactions on Medical Imaging*, 19(3), 203-210.
DOI: [10.1109/42.845178](https://doi.org/10.1109/42.845178)

```bibtex
@article{hoover2000stare,
  author  = {Hoover, A. D. and Kouznetsova, V. and Goldbaum, M.},
  title   = {Locating blood vessels in retinal images},
  journal = {IEEE Transactions on Medical Imaging},
  volume  = {19}, number = {3}, pages = {203--210},
  year    = {2000}
}
```
