"""Fine-tuning machinery tests that need neither sam2 nor checkpoints."""

import nibabel as nib
import numpy as np
import torch

from medseg_label_efficiency.finetune_sam import build_pairs, freeze_all_but_decoder, precompute


class TinyModel(torch.nn.Module):
    """Mimics sam2's naming: one decoder submodule among frozen ones."""

    def __init__(self, image_size: int = 64):
        super().__init__()
        self.image_size = image_size
        self.image_encoder = torch.nn.Linear(4, 4)
        self.sam_prompt_encoder = torch.nn.Linear(4, 4)
        self.sam_mask_decoder = torch.nn.Linear(4, 4)


class FakePredictor:
    """set_image leaves features shaped like sam2's, scaled to the model size."""

    def __init__(self, model):
        self.model = model
        self._features = None

    def set_image(self, rgb):
        s = self.model.image_size
        self._features = {
            "image_embed": torch.zeros(1, 8, s // 16, s // 16),
            "high_res_feats": [
                torch.zeros(1, 4, s // 4, s // 4), torch.zeros(1, 4, s // 8, s // 8)
            ],
        }


class FakeBackend:
    def __init__(self, image_size):
        self.predictor = FakePredictor(TinyModel(image_size))


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
        {"gt_square": gt, "patient": "patient0001"},
        {"gt_square": ~torch.zeros(3, 8, 8, dtype=torch.bool), "patient": "patient0002"},
    ]
    pairs = build_pairs(entries, (1, 2, 3))
    assert (0, 1) in pairs
    assert (0, 2) not in pairs and (0, 3) not in pairs
    assert [(1, s) for s in (1, 2, 3)] == [p for p in pairs if p[0] == 1]


def test_precompute_uses_the_model_image_size(tmp_path):
    # the GT square and the cached features follow the model, not a constant
    image = np.random.default_rng(0).random((90, 70)).astype(np.float32)
    label = np.zeros((90, 70), dtype=np.uint8)
    label[20:50, 20:50] = 1
    nib.save(nib.Nifti1Image(image, np.eye(4)), tmp_path / "img.nii.gz")
    nib.save(nib.Nifti1Image(label, np.eye(4)), tmp_path / "lbl.nii.gz")
    sample = {"image": str(tmp_path / "img.nii.gz"), "label": str(tmp_path / "lbl.nii.gz"),
              "patient": "patient0001"}
    for size in (64, 128):
        entries = precompute(FakeBackend(size), [sample], (1, 2, 3))
        assert entries[0]["gt_square"].shape == (3, size, size)
        assert entries[0]["embed"].shape == (8, size // 16, size // 16)
        assert entries[0]["embed"].dtype == torch.float16
        assert entries[0]["gt_square"][0].any() and not entries[0]["gt_square"][1].any()


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
