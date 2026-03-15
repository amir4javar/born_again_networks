"""
Unified model builder — dispatches to ResNet or EfficientNet based on the
architecture name in the config.

Usage:
    from models.builder import build_model, build_self_distill_model

    model = build_model("resnet18", num_classes=100)
    model = build_model("efficientnet_b0", num_classes=100)

    sd_model = build_self_distill_model("resnet18", num_classes=100)
    sd_model = build_self_distill_model("efficientnet_b0", num_classes=100)
"""

import torch.nn as nn

from .resnet import build_resnet, _MODELS as _RESNET_MODELS
from .efficientnet import build_efficientnet, _MODELS as _EFFICIENTNET_MODELS
from .resnet_self_distill import build_self_distill_resnet
from .efficientnet_self_distill import build_self_distill_efficientnet


_ALL_MODELS = list(_RESNET_MODELS.keys()) + list(_EFFICIENTNET_MODELS.keys())


def build_model(arch: str, num_classes: int = 100) -> nn.Module:
    """Build a standard (non-self-distill) model by architecture name."""
    if arch in _RESNET_MODELS:
        return build_resnet(arch, num_classes)
    if arch in _EFFICIENTNET_MODELS:
        return build_efficientnet(arch, num_classes)
    raise ValueError(f"Unknown architecture '{arch}'. "
                     f"Choose from {_ALL_MODELS}")


def build_self_distill_model(arch: str, num_classes: int = 100) -> nn.Module:
    """Build a self-distillation model (with aux heads) by architecture name."""
    if arch in _RESNET_MODELS:
        return build_self_distill_resnet(arch, num_classes)
    if arch in _EFFICIENTNET_MODELS:
        return build_self_distill_efficientnet(arch, num_classes)
    raise ValueError(f"Unknown architecture '{arch}'. "
                     f"Choose from {_ALL_MODELS}")
