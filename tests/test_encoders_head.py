"""Frozen-encoder registry contract and head tests — no hub downloads."""

import pytest
import torch

from medseg_label_efficiency.encoders import FROZEN_ENCODERS, FrozenEncoder
from medseg_label_efficiency.seg_head import PARAM_WINDOW, build_head


def test_encoder_registry_entries_complete():
    for name, entry in FROZEN_ENCODERS.items():
        assert {"hub", "checkpoint", "patch", "dim", "input_size"} <= set(entry), name
        assert entry["input_size"] % entry["patch"] == 0, name


def test_unknown_encoder_error_lists_known():
    with pytest.raises(KeyError, match="dinov2_s"):
        FrozenEncoder("no_such_encoder")


def test_gated_checkpoint_missing_gives_clear_error(monkeypatch):
    monkeypatch.setitem(
        FROZEN_ENCODERS,
        "gated_test",
        {"hub": ("x/y", "z"), "checkpoint": "does/not/exist.pth",
         "patch": 16, "dim": 8, "input_size": 32},
    )
    with pytest.raises(FileNotFoundError, match="license"):
        FrozenEncoder("gated_test")


def test_head_output_shape_and_capacity_window():
    head = build_head(in_dim=384, num_classes=4)
    n = sum(p.numel() for p in head.parameters())
    assert PARAM_WINDOW[0] <= n <= PARAM_WINDOW[1]
    feats = torch.randn(2, 384, 37, 37)
    out = head(feats, out_size=518)
    assert out.shape == (2, 4, 518, 518)


def test_head_trains_one_step():
    head = build_head(in_dim=384, num_classes=4)
    feats = torch.randn(2, 384, 32, 32)
    gt = torch.randint(0, 4, (2, 512, 512))
    logits = head(feats, out_size=512)
    loss = torch.nn.functional.cross_entropy(logits, gt)
    loss.backward()
    assert torch.isfinite(loss)


def test_head_checkpoint_roundtrip(tmp_path):
    head = build_head(384, 4)
    path = tmp_path / "head.pt"
    torch.save({"head": head.state_dict(), "step": 3}, path)
    restored = build_head(384, 4)
    restored.load_state_dict(torch.load(path, weights_only=True)["head"])
    a = dict(head.named_parameters())
    b = dict(restored.named_parameters())
    assert all(torch.equal(a[k], b[k]) for k in a)
