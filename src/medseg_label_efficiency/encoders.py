"""Frozen-encoder registry — the third registry, for the DINO-style arm.

A frozen encoder turns an image into a grid of patch tokens: a low-resolution
"meaning map" where every 14x14 (or 16x16) pixel patch becomes one feature
vector. The encoder is never trained here; a small head (seg_head.py) learns
to read the map. Adding an encoder = one FROZEN_ENCODERS entry.

DINOv3's weights are license-gated: the user accepts Meta's license and
places the file at the path named in the entry; DINOv2 downloads itself
through torch.hub.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

FROZEN_ENCODERS = {
    "dinov2_s": {
        "hub": ("facebookresearch/dinov2", "dinov2_vits14"),
        "checkpoint": None,  # ungated, torch.hub fetches the weights
        "patch": 14,
        "dim": 384,
        "input_size": 518,  # must be a multiple of the patch size -> 37x37 tokens
    },
    "dinov3_s": {
        "hub": ("facebookresearch/dinov3", "dinov3_vits16"),
        "checkpoint": "checkpoints/dinov3_vits16.pth",  # license-gated, manual download
        "patch": 16,
        "dim": 384,
        "input_size": 512,  # 32x32 tokens
    },
}


class FrozenEncoder:
    def __init__(self, name: str, device: str = "cpu"):
        if name not in FROZEN_ENCODERS:
            known = ", ".join(sorted(FROZEN_ENCODERS))
            raise KeyError(f"unknown frozen encoder {name!r}; registered: {known}")
        entry = FROZEN_ENCODERS[name]
        self.name = name
        self.dim = entry["dim"]
        self.input_size = entry["input_size"]
        self.device = torch.device(device)

        repo, model_name = entry["hub"]
        if entry["checkpoint"] is None:
            self.model = torch.hub.load(repo, model_name)
        else:
            ckpt = Path(entry["checkpoint"])
            if not ckpt.is_file():
                raise FileNotFoundError(
                    f"{name} weights not found at {ckpt} — these are license-gated: "
                    "accept the DINOv3 license and download the ViT-S/16 file there"
                )
            self.model = torch.hub.load(repo, model_name, weights=str(ckpt))
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def embed(self, image_rgb: np.ndarray) -> torch.Tensor:
        """uint8 (H, W, 3) -> patch-token grid (dim, Hp, Wp), on CPU."""
        x = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        x = F.interpolate(
            x[None], size=(self.input_size, self.input_size),
            mode="bilinear", align_corners=False,
        )
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        feats = self.model.get_intermediate_layers(x.to(self.device), n=1, reshape=True)[0]
        return feats[0].cpu()
