"""
EfficientNet for CIFAR (32×32 inputs).

Implements EfficientNet-B0, B1, and B2 adapted for CIFAR's small spatial
resolution. Uses the standard MBConv inverted-residual blocks with squeeze-
and-excitation, but replaces the ImageNet stem (stride-2) with a stride-1
3×3 conv so feature maps don't shrink too aggressively on 32×32 inputs.

The network is organized into 7 "stages" (groups of MBConv blocks).  For
self-distillation we expose 4 feature-extraction points that map onto the
same layer1–layer4 interface used by the ResNet models:
    layer1 → after stages 0–1   (early, low-level features)
    layer2 → after stages 2–3
    layer3 → after stages 4–5
    layer4 → after stage 6      (deep, high-level features)
"""

import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Configuration per variant ────────────────────────────────────────────────

@dataclass
class _BlockCfg:
    expand_ratio: int
    channels: int
    num_layers: int
    stride: int
    kernel_size: int


def _scale_cfg(base_cfgs: List[_BlockCfg], width_mult: float,
               depth_mult: float) -> List[_BlockCfg]:
    """Scale channel widths and layer depths by the given multipliers."""
    scaled = []
    for cfg in base_cfgs:
        scaled.append(_BlockCfg(
            expand_ratio=cfg.expand_ratio,
            channels=int(math.ceil(cfg.channels * width_mult / 8) * 8),
            num_layers=int(math.ceil(cfg.num_layers * depth_mult)),
            stride=cfg.stride,
            kernel_size=cfg.kernel_size,
        ))
    return scaled


# EfficientNet-B0 base configuration (7 stages)
_BASE_CFGS = [
    _BlockCfg(1, 16, 1, 1, 3),   # stage 0
    _BlockCfg(6, 24, 2, 2, 3),   # stage 1
    _BlockCfg(6, 40, 2, 2, 5),   # stage 2
    _BlockCfg(6, 80, 3, 2, 3),   # stage 3
    _BlockCfg(6, 112, 3, 1, 5),  # stage 4
    _BlockCfg(6, 192, 4, 2, 5),  # stage 5
    _BlockCfg(6, 320, 1, 1, 3),  # stage 6
]

# (width_multiplier, depth_multiplier)
_VARIANT_PARAMS = {
    "efficientnet_b0": (1.0, 1.0),
    "efficientnet_b1": (1.0, 1.1),
    "efficientnet_b2": (1.1, 1.2),
}


# ── Building blocks ─────────────────────────────────────────────────────────

class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels, se_ratio=0.25):
        super().__init__()
        squeezed = max(1, int(channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, squeezed, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeezed, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv (MBConv) with optional SE."""

    def __init__(self, in_channels, out_channels, expand_ratio,
                 kernel_size=3, stride=1, se_ratio=0.25):
        super().__init__()
        mid_channels = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []
        # Expansion phase (skip if expand_ratio == 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True),
            ])

        # Depthwise conv
        layers.extend([
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        ])

        # Squeeze-and-Excitation
        layers.append(SqueezeExcite(mid_channels, se_ratio))

        # Pointwise linear projection
        layers.extend([
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


# ── EfficientNet backbone ───────────────────────────────────────────────────

class EfficientNet(nn.Module):
    """EfficientNet adapted for CIFAR-sized (32×32) inputs.

    Exposes the same external interface as the ResNet class:
        - forward(x) → logits
        - forward_features(x) → ([f1, f2, f3, f4], logits)
        - layer1 … layer4 attributes (nn.Sequential groups)
    """

    def __init__(self, block_cfgs: List[_BlockCfg], num_classes: int = 100):
        super().__init__()

        # Stem: 3×3, stride-1 for CIFAR (not stride-2 like ImageNet)
        stem_channels = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
        )

        # Build the 7 stages of MBConv blocks
        stages = []
        in_ch = stem_channels
        for cfg in block_cfgs:
            blocks = []
            for i in range(cfg.num_layers):
                stride = cfg.stride if i == 0 else 1
                blocks.append(MBConv(in_ch, cfg.channels, cfg.expand_ratio,
                                     cfg.kernel_size, stride))
                in_ch = cfg.channels
            stages.append(nn.Sequential(*blocks))

        # Group 7 stages into 4 "layers" for compatibility with ResNet interface
        # layer1 = stages 0–1, layer2 = stages 2–3,
        # layer3 = stages 4–5, layer4 = stage 6
        self.layer1 = nn.Sequential(*stages[0:2])
        self.layer2 = nn.Sequential(*stages[2:4])
        self.layer3 = nn.Sequential(*stages[4:6])
        self.layer4 = nn.Sequential(*stages[6:7])

        # Record output channels for each layer group (needed by aux heads)
        self.layer_channels = [
            block_cfgs[1].channels,  # end of stage 1
            block_cfgs[3].channels,  # end of stage 3
            block_cfgs[5].channels,  # end of stage 5
            block_cfgs[6].channels,  # end of stage 6
        ]

        # Head
        head_channels = block_cfgs[-1].channels * 4  # 1280 for B0
        self.head_conv = nn.Sequential(
            nn.Conv2d(block_cfgs[-1].channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(head_channels, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.head_conv(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

    def forward_features(self, x):
        """Return intermediate feature maps for self-distillation."""
        out = self.stem(x)
        f1 = self.layer1(out)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        out = self.head_conv(f4)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        logits = self.fc(out)
        return [f1, f2, f3, f4], logits


# ── Convenience constructors ─────────────────────────────────────────────────

def _build_efficientnet(variant: str, num_classes: int = 100) -> EfficientNet:
    width_mult, depth_mult = _VARIANT_PARAMS[variant]
    cfgs = _scale_cfg(_BASE_CFGS, width_mult, depth_mult)
    return EfficientNet(cfgs, num_classes)


def efficientnet_b0(num_classes=100):
    return _build_efficientnet("efficientnet_b0", num_classes)


def efficientnet_b1(num_classes=100):
    return _build_efficientnet("efficientnet_b1", num_classes)


def efficientnet_b2(num_classes=100):
    return _build_efficientnet("efficientnet_b2", num_classes)


_MODELS = {
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b1": efficientnet_b1,
    "efficientnet_b2": efficientnet_b2,
}


def build_efficientnet(name: str, num_classes: int = 100) -> EfficientNet:
    if name not in _MODELS:
        raise ValueError(f"Unknown architecture '{name}'. "
                         f"Choose from {list(_MODELS.keys())}")
    return _MODELS[name](num_classes)
