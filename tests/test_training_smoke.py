"""Wiring smoke tests for the training stack. CPU-only, no dataset needed."""

from pathlib import Path

import torch
from monai.losses import DiceCELoss

from medseg_label_efficiency.config import load_config
from medseg_label_efficiency.train import build_model, pick_device

CONFIG = Path(__file__).parents[1] / "configs" / "camus_unet2d.yaml"


def test_model_forward_backward():
    cfg = load_config(CONFIG)
    model = build_model(cfg, 4)
    images = torch.randn(2, 1, 256, 256)
    labels = torch.randint(0, 4, (2, 1, 256, 256))
    out = model(images)
    assert out.shape == (2, 4, 256, 256)
    loss = DiceCELoss(to_onehot_y=True, softmax=True)(out, labels)
    loss.backward()
    assert torch.isfinite(loss)


def test_pick_device_explicit():
    assert pick_device("cpu").type == "cpu"
