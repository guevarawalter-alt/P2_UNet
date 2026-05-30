# -*- coding: utf-8 -*-
"""
src/model.py
============
Arquitectura U-Net implementada desde cero en PyTorch.
Sin copiar de librerías de segmentación existentes (segmentation_models_pytorch, MONAI, etc.).

Pregunta 2 — Examen Parcial RNAPD 2026-I
Docente: Ph.D. Aldo Camargo

Referencia:
    Ronneberger, O., Fischer, P., & Brox, T. (2015).
    U-Net: Convolutional networks for biomedical image segmentation.
    MICCAI 2015, pp. 234-241.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Bloque convolucional doble ────────────────────────────────────────────
def double_conv(in_ch: int, out_ch: int,
                dropout_p: float = 0.0) -> nn.Sequential:
    """
    Conv(3×3) + BN + ReLU + Conv(3×3) + BN + ReLU [+ Dropout2d]

    bias=False porque BatchNorm ya tiene parámetro beta que cumple
    la misma función; evitar redundancia mejora la convergencia.
    """
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout_p > 0:
        layers.append(nn.Dropout2d(p=dropout_p))
    return nn.Sequential(*layers)


# ── Bloque Encoder ────────────────────────────────────────────────────────
class EncoderBlock(nn.Module):
    """
    Bloque del encoder: Conv doble + MaxPool(2×2).
    Devuelve (salida_comprimida, skip_connection).

    La skip_connection se almacena para pasarla al decoder correspondiente.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch, dropout_p)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return self.pool(skip), skip


# ── Bloque Decoder — Concatenación (U-Net original) ───────────────────────
class DecoderBlockConcat(nn.Module):
    """
    Decoder con skip por CONCATENACIÓN — U-Net original.
    Ventaja: preserva independientemente los canales del encoder
             y del decoder; la conv decide cómo ponderarlos.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch,
                                        kernel_size=2, stride=2)
        self.conv = double_conv(in_ch, out_ch, dropout_p)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


# ── Bloque Decoder — Suma (variante para ablación) ───────────────────────
class DecoderBlockSum(nn.Module):
    """
    Decoder con skip por SUMA elemento a elemento — variante ResNet-style.
    Limitación: puede cancelar gradientes de signo opuesto.
    Usado en el estudio de ablación para comparar con concatenación.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch,
                                        kernel_size=2, stride=2)
        self.proj = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)
        self.conv = double_conv(out_ch, out_ch, dropout_p)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
        return self.conv(x + self.proj(skip))


# ── U-Net completa ────────────────────────────────────────────────────────
class UNet(nn.Module):
    """
    U-Net para segmentación binaria de vasos retinianos.

    Arquitectura:
        Encoder:     4 bloques, reduce 512px → 32px (factor 1/16)
        Bottleneck:  1024 canales, 32×32px, Dropout(dropout_p)
        Decoder:     4 bloques, recupera 32px → 512px
        Cabeza:      Conv(1×1) → logit binario por píxel

    Parámetros
    ----------
    in_channels : canales de entrada (3 = RGB)
    num_classes : canales de salida (1 = segmentación binaria)
    features    : canales por nivel del encoder
    skip_mode   : 'concat' (U-Net original) | 'sum' (variante ablación)
    dropout_p   : dropout en niveles profundos del encoder y decoder

    Total de parámetros: ~31,037,633 (con features=(64,128,256,512))
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        features: tuple = (64, 128, 256, 512),
        skip_mode: str = "concat",
        dropout_p: float = 0.2,
    ):
        super().__init__()
        DecoderBlock = (DecoderBlockConcat if skip_mode == "concat"
                        else DecoderBlockSum)

        # Encoder
        self.encoders = nn.ModuleList()
        ch = in_channels
        for i, f in enumerate(features):
            dp = dropout_p if i >= 2 else 0.0  # dropout solo en niveles 3 y 4
            self.encoders.append(EncoderBlock(ch, f, dp))
            ch = f

        # Bottleneck
        self.bottleneck = double_conv(ch, ch * 2, dropout_p)
        ch = ch * 2

        # Decoder
        self.decoders = nn.ModuleList()
        for i, f in enumerate(reversed(features)):
            dp = dropout_p if i < 2 else 0.0   # dropout solo en niveles 4 y 3
            self.decoders.append(DecoderBlock(ch, f, dp))
            ch = f

        # Cabeza de segmentación binaria
        self.head = nn.Conv2d(ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.head(x)   # (B, 1, H, W) logits sin sigmoid


def count_params(model: nn.Module) -> int:
    """Cuenta los parámetros entrenables del modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
