"""
EfficientNet with auxiliary classifier heads for layer-wise self-distillation.

Same design as ResNetSelfDistill: three auxiliary heads attached after
layer1, layer2, and layer3.  The deepest classifier (main head after layer4)
acts as the teacher during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .efficientnet import EfficientNet
from .resnet_self_distill import AuxiliaryHead  # reuse the same head design


class EfficientNetSelfDistill(nn.Module):
    """EfficientNet with auxiliary heads for single-pass self-distillation.

    Returns
    -------
    logits_main : Tensor   – final classifier output (teacher)
    aux_logits  : list[Tensor] – [aux1, aux2, aux3] from shallow → deep
    """

    def __init__(self, base_model: EfficientNet, num_classes: int = 100):
        super().__init__()
        self.stem = base_model.stem
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.head_conv = base_model.head_conv
        self.avgpool = base_model.avgpool
        self.fc = base_model.fc

        # Channel widths from the base model
        ch1, ch2, ch3, _ = base_model.layer_channels

        self.aux1 = AuxiliaryHead(ch1, num_classes, inner_channels=256)
        self.aux2 = AuxiliaryHead(ch2, num_classes, inner_channels=256)
        self.aux3 = AuxiliaryHead(ch3, num_classes, inner_channels=256)

    def forward(self, x):
        out = self.stem(x)
        f1 = self.layer1(out)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        out = self.head_conv(f4)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        logits_main = self.fc(out)

        aux1_logits = self.aux1(f1)
        aux2_logits = self.aux2(f2)
        aux3_logits = self.aux3(f3)

        return logits_main, [aux1_logits, aux2_logits, aux3_logits]

    def forward_main_only(self, x):
        """Inference mode — only the main classifier."""
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.head_conv(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def build_self_distill_efficientnet(arch: str, num_classes: int = 100):
    """Build an EfficientNetSelfDistill model from an architecture name."""
    from .efficientnet import build_efficientnet
    base = build_efficientnet(arch, num_classes)
    return EfficientNetSelfDistill(base, num_classes)
