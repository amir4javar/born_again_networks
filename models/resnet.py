"""
ResNet for CIFAR (32×32 inputs).

Standard torchvision ResNets expect 224×224 ImageNet inputs.  This module
provides ResNet-18 / 34 / 50 variants adapted for CIFAR's smaller spatial
resolution: the first conv uses 3×3 / stride-1 (no 7×7 / stride-2 + maxpool)
so feature maps don't shrink too aggressively.

Each model exposes four "layer groups" (layer1 … layer4) whose outputs are
used as attachment points for auxiliary heads in the self-distillation variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ──────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


# ── ResNet backbone ──────────────────────────────────────────────────────────

class ResNet(nn.Module):
    """ResNet adapted for CIFAR-sized (32×32) inputs."""

    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_planes = 64

        # CIFAR stem: 3×3 conv, no maxpool
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def forward_features(self, x):
        """Return intermediate feature maps for self-distillation."""
        out = F.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(out)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        pooled = self.avgpool(f4)
        pooled = torch.flatten(pooled, 1)
        logits = self.fc(pooled)
        return [f1, f2, f3, f4], logits


# ── Convenience constructors ─────────────────────────────────────────────────

def resnet18(num_classes=100):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=100):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes=100):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


_MODELS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}


def build_resnet(name: str, num_classes: int = 100) -> ResNet:
    if name not in _MODELS:
        raise ValueError(f"Unknown architecture '{name}'. "
                         f"Choose from {list(_MODELS.keys())}")
    return _MODELS[name](num_classes)
