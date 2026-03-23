"""Minimal helpers for pretrained ResNet-18 transfer runs."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_pretrained_resnet18(
    num_classes: int,
    transfer_mode: str,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a pretrained ResNet-18 configured for CIFAR-10 transfer learning."""

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    if transfer_mode == "modify_finetune":
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
        model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if transfer_mode == "resize_freeze":
        for parameter in model.parameters():
            parameter.requires_grad = False
        if freeze_backbone:
            for parameter in model.fc.parameters():
                parameter.requires_grad = True
        else:
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True
            for parameter in model.fc.parameters():
                parameter.requires_grad = True

    return model
