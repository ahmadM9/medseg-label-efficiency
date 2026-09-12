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
        "kind": "hub",  # torch.hub fetches code and (ungated) weights itself
        "source": ("facebookresearch/dinov2", "dinov2_vits14"),
        "patch": 14,
        "dim": 384,
        "input_size": 518,  # must be a multiple of the patch size -> 37x37 tokens
    },
    "dinov3_s": {
        # license-gated: the HuggingFace snapshot is downloaded manually into
        # this directory (transformers format), then loaded offline
        "kind": "transformers",
        "source": "checkpoints/dinov3_vits16_hf",
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
        self.kind = entry["kind"]
        self.dim = entry["dim"]
        self.input_size = entry["input_size"]
        self.device = torch.device(device)
        self._skip_tokens = 0  # transformers models prepend cls/register tokens

        if self.kind == "hub":
            repo, model_name = entry["source"]
            self.model = torch.hub.load(repo, model_name)
        elif self.kind == "transformers":
            snapshot = Path(entry["source"])
            if not (snapshot / "model.safetensors").is_file():
                raise FileNotFoundError(
                    f"{name} weights not found at {snapshot} — these are license-gated: "
                    "accept the license on HuggingFace and download the snapshot there"
                )
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(snapshot)
            self._skip_tokens = 1 + getattr(self.model.config, "num_register_tokens", 0)
        else:
            raise ValueError(f"unknown encoder kind {self.kind!r}")
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
        x = x.to(self.device)
        if self.kind == "hub":
            feats = self.model.get_intermediate_layers(x, n=1, reshape=True)[0]
            return feats[0].cpu()
        # transformers: token sequence -> drop cls/register tokens -> grid
        tokens = self.model(pixel_values=x).last_hidden_state[0, self._skip_tokens :]
        side = self.input_size // FROZEN_ENCODERS[self.name]["patch"]
        return tokens.reshape(side, side, self.dim).permute(2, 0, 1).cpu()
