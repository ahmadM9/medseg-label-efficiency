"""Harness tests for the promptable-model evaluation, using a mock backend
so neither the sam2 package nor its checkpoints are needed (e.g. in CI)."""

import nibabel as nib
import numpy as np
import pytest

from medseg_label_efficiency.eval_sam import evaluate_model, summarize, to_rgb_uint8


class MockBackend:
    """Fills the box, or draws a disk around the point — imperfect on purpose."""

    def predict(self, image_rgb, box=None, point=None):
        mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        if box is not None:
            r0, c0, r1, c1 = box
            mask[r0 : r1 + 1, c0 : c1 + 1] = True
        elif point is not None:
            yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
            mask[(yy - point[0]) ** 2 + (xx - point[1]) ** 2 <= 15**2] = True
        return mask


@pytest.fixture
def synthetic_samples(tmp_path):
    image = np.random.default_rng(0).random((120, 100)).astype(np.float32) * 200
    label = np.zeros((120, 100), dtype=np.uint8)
    label[30:60, 30:60] = 1
    label[20:30, 30:60] = 2
    label[70:95, 35:65] = 3
    affine = np.diag([0.5, 0.4, 1.0, 1.0])  # anisotropic mm spacing
    nib.save(nib.Nifti1Image(image, affine), tmp_path / "img.nii.gz")
    nib.save(nib.Nifti1Image(label, affine), tmp_path / "lbl.nii.gz")
    return [
        {
            "image": str(tmp_path / "img.nii.gz"),
            "label": str(tmp_path / "lbl.nii.gz"),
            "patient": "patient0001",
            "view": "2CH",
            "phase": "ED",
            "quality": "Good",
        }
    ]


@pytest.mark.parametrize("prompt_type", ["box", "point"])
def test_evaluate_model_record_format(synthetic_samples, prompt_type):
    records = evaluate_model(MockBackend(), synthetic_samples, prompt_type)
    assert len(records) == 1
    row = records[0]
    for name in ("lv_endo", "lv_myo", "left_atrium"):
        assert 0.0 <= row[f"dice_{name}"] <= 1.0
        assert row[f"hd95_{name}"] >= 0.0 or np.isnan(row[f"hd95_{name}"])
    assert row["quality"] == "Good"


def test_box_mock_scores_reasonably(synthetic_samples):
    # a filled box over a solid square should overlap it heavily
    records = evaluate_model(MockBackend(), synthetic_samples, "box")
    assert records[0]["dice_lv_endo"] > 0.7


def test_summarize_shapes(synthetic_samples):
    records = evaluate_model(MockBackend(), synthetic_samples, "box")
    summary = summarize(records)
    assert "dice_mean" in summary and "hd95_mean" in summary
    assert summary["by_quality"]["Good"]["n"] == 1


def test_to_rgb_uint8_range():
    rgb = to_rgb_uint8(np.array([[0.0, 512.0], [128.0, 256.0]]))
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert rgb.max() == 255 and rgb.min() == 0
