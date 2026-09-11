"""The small trainable segmentation head for frozen-encoder arms.

Reads a patch-token grid (dim, Hp, Wp) and produces full-resolution
multi-class logits. Capacity is deliberately matched to MedSAM2's mask
decoder (~4.2M trainable params) so the two adaptation arms train a
comparable amount of new capacity — the benchmark's fairness anchor.
"""

import torch
import torch.nn.functional as F
from torch import nn

# widths chosen so a 384-dim input lands in the 3.5-4.5M parameter window
WIDTHS = (512, 288, 144, 72)
PARAM_WINDOW = (3.5e6, 4.5e6)


class SegHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        blocks, width_in = [], in_dim
        for width in WIDTHS:
            blocks += [
                nn.Conv2d(width_in, width, 3, padding=1),
                nn.GroupNorm(8, width),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ]
            width_in = width
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Conv2d(width_in, num_classes, 1)

    def forward(self, feats: torch.Tensor, out_size: int) -> torch.Tensor:
        logits = self.classifier(self.blocks(feats))
        return F.interpolate(logits, size=(out_size, out_size), mode="bilinear")


def build_head(in_dim: int, num_classes: int) -> SegHead:
    head = SegHead(in_dim, num_classes)
    n_params = sum(p.numel() for p in head.parameters())
    low, high = PARAM_WINDOW
    assert low <= n_params <= high, (
        f"head has {n_params / 1e6:.2f}M params, outside the capacity-matching "
        f"window [{low / 1e6:.1f}M, {high / 1e6:.1f}M] — adjust WIDTHS"
    )
    print(f"trainable head parameters: {n_params / 1e6:.2f}M")
    return head
