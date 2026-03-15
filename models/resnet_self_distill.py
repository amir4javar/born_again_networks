"""
ResNet with auxiliary classifier heads for layer-wise self-distillation
(BYOT — "Be Your Own Teacher", Zhang et al. 2019).

Three auxiliary classifiers are attached after layer1, layer2, and layer3.
Each auxiliary head consists of:
  1. A bottleneck block to reduce channels
  2. Adaptive average pooling to 1×1
  3. A linear classifier

During training the deepest classifier (the original head at layer4) acts as
the teacher, and the three shallower classifiers are students.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock, Bottleneck, ResNet


class AuxiliaryHead(nn.Module):
    """Lightweight auxiliary classifier attached to an intermediate feature."""

    def __init__(self, in_channels, num_classes, inner_channels=512):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inner_channels, num_classes)

    def forward(self, x):
        out = self.block(x)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


class ResNetSelfDistill(nn.Module):
    """ResNet with auxiliary heads for single-pass self-distillation.

    Returns
    -------
    logits_main : Tensor   – final classifier output (teacher)
    aux_logits  : list[Tensor] – [aux1, aux2, aux3] from shallow → deep
    """

    def __init__(self, base_model: ResNet, num_classes: int = 100):
        super().__init__()
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        self.fc = base_model.fc

        # Determine channel widths after each layer
        block = type(list(base_model.layer1.children())[0])
        expansion = block.expansion
        ch1 = 64 * expansion
        ch2 = 128 * expansion
        ch3 = 256 * expansion

        self.aux1 = AuxiliaryHead(ch1, num_classes, inner_channels=256)
        self.aux2 = AuxiliaryHead(ch2, num_classes, inner_channels=256)
        self.aux3 = AuxiliaryHead(ch3, num_classes, inner_channels=256)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(out)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        pooled = self.avgpool(f4)
        pooled = torch.flatten(pooled, 1)
        logits_main = self.fc(pooled)

        aux1_logits = self.aux1(f1)
        aux2_logits = self.aux2(f2)
        aux3_logits = self.aux3(f3)

        return logits_main, [aux1_logits, aux2_logits, aux3_logits]

    def forward_main_only(self, x):
        """Inference mode — only the main classifier."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def build_self_distill_resnet(arch: str, num_classes: int = 100):
    """Build a ResNetSelfDistill model from an architecture name."""
    from .resnet import build_resnet
    base = build_resnet(arch, num_classes)
    return ResNetSelfDistill(base, num_classes)
