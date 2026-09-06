"""Transform-chain tests on synthetic NIfTI files — no real data needed."""

import nibabel as nib
import numpy as np
import pytest
import torch

from medseg_label_efficiency.data.camus import get_transforms


@pytest.fixture
def synthetic_sample(tmp_path):
    """A fake echo-like image/label pair with CAMUS-style intensity range."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(384, 512)).astype(np.float32)
    label = np.zeros((384, 512), dtype=np.uint8)
    label[100:200, 150:250] = 1
    label[80:100, 150:250] = 2
    label[210:260, 160:240] = 3
    affine = np.eye(4)
    image_path = tmp_path / "img.nii.gz"
    label_path = tmp_path / "lbl.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(label, affine), label_path)
    return {"image": str(image_path), "label": str(label_path)}


def test_val_transforms_shapes_and_range(synthetic_sample):
    out = get_transforms(train=False)(synthetic_sample)
    assert out["image"].shape == (1, 256, 256)
    assert out["label"].shape == (1, 256, 256)
    assert out["image"].min() >= 0.0 and out["image"].max() <= 1.0


def test_label_values_preserved_by_resize(synthetic_sample):
    out = get_transforms(train=False)(synthetic_sample)
    values = set(torch.unique(out["label"]).tolist())
    assert values <= {0, 1, 2, 3}
    # nearest-neighbour resize must not invent fractional labels
    assert all(float(v).is_integer() for v in values)


def test_train_transforms_shapes(synthetic_sample):
    out = get_transforms(train=True)(synthetic_sample)
    assert out["image"].shape == (1, 256, 256)
    assert out["label"].shape == (1, 256, 256)
    assert set(torch.unique(out["label"]).tolist()) <= {0, 1, 2, 3}


def test_custom_image_size(synthetic_sample):
    out = get_transforms(train=False, image_size=(128, 128))(synthetic_sample)
    assert out["image"].shape == (1, 128, 128)
