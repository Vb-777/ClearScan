"""ResNet-18 for 7 skin-lesion classes. Images are 28x28; they are resized to 64x64 inside the
model so the deployed model takes the dataset's native size."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class SkinNet(nn.Module):
    def __init__(self, n_classes: int = 7, pretrained: str | Path | None = None, size: int = 64):
        super().__init__()
        self.size = size
        self.net = torchvision.models.resnet18(num_classes=1000)
        if pretrained:
            # ImageNet weights (timm "ResNet strikes back" A1 recipe), keys match torchvision's ResNet-18
            self.net.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=True)
        self.net.fc = nn.Linear(512, n_classes)
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def forward(self, x_uint8_nchw: torch.Tensor) -> torch.Tensor:
        """x: N x 3 x 28 x 28, values 0-255 (float or uint8). Returns logits."""
        x = x_uint8_nchw.float() / 255.0
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std)


def to_nchw(x_nhwc_uint8) -> torch.Tensor:
    return torch.as_tensor(x_nhwc_uint8).permute(0, 3, 1, 2).contiguous()
