"""Harness tests for the promptable-model evaluation, using a mock backend
so neither the sam2 package nor its checkpoints are needed (e.g. in CI)."""

import nibabel as nib
import numpy as np
import pytest

from medseg_label_efficiency.data.registry import get_dataset_module
from medseg_label_efficiency.eval_sam import evaluate_model, summarize, to_rgb_uint8
from medseg_label_efficiency.reporting import combine_binary_masks, load_mask, save_mask

CAMUS = get_dataset_module("camus")


class MockBackend:
    """Fills the box, or draws a disk around the point, imperfect on purpose;
    remembers every prompt it was given."""

    def __init__(self):
        self.boxes = []

    def predict(self, image_rgb, box=None, point=None):
        mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        if box is not None:
            self.boxes.append(box)
            r0, c0, r1, c1 = box
            mask[r0 : r1 + 1, c0 : c1 + 1] = True
        elif point is not None:
            yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
            mask[(yy - point[0]) ** 2 + (xx - point[1]) ** 2 <= 15**2] = True
        return mask


def make_sample(tmp_path, phase: str) -> dict:
    image = np.random.default_rng(0).random((120, 100)).astype(np.float32) * 200
    label = np.zeros((120, 100), dtype=np.uint8)
    label[30:60, 30:60] = 1
    label[20:30, 30:60] = 2
    label[70:95, 35:65] = 3
    affine = np.diag([0.5, 0.4, 1.0, 1.0])  # anisotropic mm spacing
    nib.save(nib.Nifti1Image(image, affine), tmp_path / f"img_{phase}.nii.gz")
    nib.save(nib.Nifti1Image(label, affine), tmp_path / f"lbl_{phase}.nii.gz")
    return {
        "image": str(tmp_path / f"img_{phase}.nii.gz"),
        "label": str(tmp_path / f"lbl_{phase}.nii.gz"),
        "patient": "patient0001",
        "view": "2CH",
        "phase": phase,
        "quality": "Good",
    }


@pytest.fixture
def synthetic_samples(tmp_path):
    return [make_sample(tmp_path, "ED")]


@pytest.mark.parametrize("prompt_type", ["box", "point"])
def test_evaluate_model_record_format(synthetic_samples, prompt_type):
    records = evaluate_model(MockBackend(), synthetic_samples, prompt_type, CAMUS)
    assert len(records) == 1
    row = records[0]
    for name in ("lv_endo", "lv_myo", "left_atrium"):
        assert 0.0 <= row[f"dice_{name}"] <= 1.0
        assert row[f"hd95_{name}"] >= 0.0 or np.isnan(row[f"hd95_{name}"])
    assert row["quality"] == "Good"
    assert 0.0 <= row["overlap_frac"] <= 1.0


def test_box_mock_scores_reasonably(synthetic_samples):
    # a filled box over a solid square should overlap it heavily
    records = evaluate_model(MockBackend(), synthetic_samples, "box", CAMUS)
    assert records[0]["dice_lv_endo"] > 0.7


def test_summarize_shapes(synthetic_samples):
    records = evaluate_model(MockBackend(), synthetic_samples, "box", CAMUS)
    summary = summarize(records, CAMUS)
    assert "dice_mean" in summary and "hd95_mean" in summary
    assert "overlap_mean" in summary
    assert summary["by_quality"]["Good"]["n"] == 1


def test_jitter_is_seeded_per_image(tmp_path):
    # the same patient's ED and ES frames must not share jitter offsets
    backend = MockBackend()
    samples = [make_sample(tmp_path, "ED"), make_sample(tmp_path, "ES")]
    evaluate_model(backend, samples, "box", CAMUS)
    ed, es = backend.boxes[:3], backend.boxes[3:]
    assert ed != es
    again = MockBackend()
    evaluate_model(again, [make_sample(tmp_path, "ED")], "box", CAMUS)
    assert again.boxes == ed


def test_masks_saved_and_reloaded(synthetic_samples, tmp_path):
    out_dir = tmp_path / "arm"
    evaluate_model(MockBackend(), synthetic_samples, "box", CAMUS, save_dir=out_dir)
    mask = load_mask(out_dir, synthetic_samples[0])
    assert mask.dtype == np.uint8 and mask.shape == (120, 100)
    assert set(np.unique(mask)) <= {0, 1, 2, 3}
    assert (mask == 3).sum() > 0


def test_cascade_boxes_come_from_source_masks(synthetic_samples, tmp_path):
    # a source that predicted only the left atrium, shifted from the GT
    source = tmp_path / "source"
    label_map = np.zeros((120, 100), dtype=np.uint8)
    label_map[75:100, 40:70] = 3
    save_mask(source, synthetic_samples[0], label_map)
    backend = MockBackend()
    records = evaluate_model(
        backend, synthetic_samples, "box", CAMUS, jitter_px=5, box_source=source
    )
    assert backend.boxes == [(75, 40, 99, 69)]  # tight box on the source, no jitter
    row = records[0]
    assert row["dice_lv_endo"] == 0.0 and row["dice_lv_myo"] == 0.0  # no prompt, empty mask
    assert 0.0 < row["dice_left_atrium"] < 1.0


def test_combine_binary_masks_lowest_id_wins():
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[:2] = True
    b[1:3] = True
    label_map, overlap = combine_binary_masks({1: a, 3: b})
    assert label_map[0, 0] == 1 and label_map[1, 0] == 1 and label_map[2, 0] == 3
    assert overlap == pytest.approx(4 / 16)


def test_to_rgb_uint8_range():
    rgb = to_rgb_uint8(np.array([[0.0, 512.0], [128.0, 256.0]]))
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert rgb.max() == 255 and rgb.min() == 0
