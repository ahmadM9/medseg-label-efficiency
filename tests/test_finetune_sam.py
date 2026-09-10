"""Fine-tuning machinery tests that need neither sam2 nor checkpoints."""

import torch

from medseg_label_efficiency.finetune_sam import build_pairs, freeze_all_but_decoder


class TinyModel(torch.nn.Module):
    """Mimics sam2's naming: one decoder submodule among frozen ones."""

    def __init__(self):
        super().__init__()
        self.image_encoder = torch.nn.Linear(4, 4)
        self.sam_prompt_encoder = torch.nn.Linear(4, 4)
        self.sam_mask_decoder = torch.nn.Linear(4, 4)


def test_only_decoder_params_train():
    model = TinyModel()
    n = freeze_all_but_decoder(model)
    assert n == sum(p.numel() for p in model.sam_mask_decoder.parameters())
    assert all(p.requires_grad for p in model.sam_mask_decoder.parameters())
    assert not any(p.requires_grad for p in model.image_encoder.parameters())
    assert not any(p.requires_grad for p in model.sam_prompt_encoder.parameters())


def test_build_pairs_skips_absent_structures():
    gt = torch.zeros(3, 8, 8, dtype=torch.bool)
    gt[0, 2:5, 2:5] = True  # only structure 1 present
    entries = [
        {"gt512": gt, "patient": "patient0001"},
        {"gt512": ~torch.zeros(3, 8, 8, dtype=torch.bool), "patient": "patient0002"},
    ]
    pairs = build_pairs(entries)
    assert (0, 1) in pairs
    assert (0, 2) not in pairs and (0, 3) not in pairs
    assert [(1, s) for s in (1, 2, 3)] == [p for p in pairs if p[0] == 1]


def test_decoder_checkpoint_roundtrip(tmp_path):
    model = TinyModel()
    path = tmp_path / "decoder.pt"
    torch.save({"decoder": model.sam_mask_decoder.state_dict(), "step": 7}, path)
    restored = TinyModel()
    ckpt = torch.load(path, weights_only=True)
    restored.sam_mask_decoder.load_state_dict(ckpt["decoder"])
    assert torch.equal(
        restored.sam_mask_decoder.weight, model.sam_mask_decoder.weight
    )
    assert ckpt["step"] == 7
