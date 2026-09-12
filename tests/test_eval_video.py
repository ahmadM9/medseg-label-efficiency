"""Harness tests for the video-mode evaluation, with mock backends so neither
sam2 nor its checkpoints are needed in CI."""

import nibabel as nib
import numpy as np
import pytest

from medseg_label_efficiency.data.camus import DERIVED_STRUCTURES, build_sequences
from medseg_label_efficiency.data.registry import get_dataset_module
from medseg_label_efficiency.eval_video import evaluate_video, structure_masks, summarize
from test_eval_sam import MockBackend

CAMUS = get_dataset_module("camus")
N_FRAMES = 3


class MockVideoBackend:
    """Fills every prompted box on every frame, ignoring the image."""

    def track(self, frames_rgb, boxes, prompt_frame):
        shape = frames_rgb[0].shape[:2]
        out = {}
        for t in range(len(frames_rgb)):
            out[t] = {}
            for obj_id, (r0, c0, r1, c1) in boxes.items():
                mask = np.zeros(shape, dtype=bool)
                mask[r0 : r1 + 1, c0 : c1 + 1] = True
                out[t][obj_id] = mask
        return out


def _label_frame(shift: int) -> np.ndarray:
    # a cavity blob (1) with a wall (2) wrapped around three sides, and an atrium (3)
    label = np.zeros((120, 100), dtype=np.uint8)
    label[30 + shift : 60 + shift, 30:60] = 2
    label[35 + shift : 60 + shift, 35:55] = 1
    label[70 + shift : 95 + shift, 35:65] = 3
    return label


@pytest.fixture
def synthetic_root(tmp_path):
    """A one-patient CAMUS-like tree with a half sequence for each view."""
    patient = "patient0001"
    (tmp_path / "database_split").mkdir()
    (tmp_path / "database_split" / "subgroup_testing.txt").write_text(f"{patient}\n")
    pdir = tmp_path / "database_nifti" / patient
    pdir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    affine = np.diag([0.5, 0.4, 1.0, 1.0])
    for view in ("2CH", "4CH"):
        (pdir / f"Info_{view}.cfg").write_text(f"ImageQuality: Good\nNbFrame: {N_FRAMES}\n")
        image = rng.random((120, 100, N_FRAMES)).astype(np.float32) * 200
        label = np.stack([_label_frame(t) for t in range(N_FRAMES)], axis=-1)
        nib.save(nib.Nifti1Image(image, affine), pdir / f"{patient}_{view}_half_sequence.nii.gz")
        nib.save(
            nib.Nifti1Image(label, affine), pdir / f"{patient}_{view}_half_sequence_gt.nii.gz"
        )
    return tmp_path


def test_build_sequences_finds_pairs(synthetic_root):
    sequences = build_sequences(synthetic_root, "test")
    assert [s["view"] for s in sequences] == ["2CH", "4CH"]
    assert all(s["n_frames"] == N_FRAMES and s["quality"] == "Good" for s in sequences)


def test_derived_structure_is_union_of_labels():
    gt = _label_frame(0)
    masks = structure_masks(gt, CAMUS)
    assert DERIVED_STRUCTURES["lv_epi"] == (1, 2)
    assert np.array_equal(masks["lv_epi"], (gt == 1) | (gt == 2))
    assert masks["lv_epi"].sum() > masks["lv_myo"].sum()


@pytest.mark.parametrize("mode", ["video", "still"])
def test_evaluate_video_records(synthetic_root, mode):
    sequences = build_sequences(synthetic_root, "test")
    backend = MockVideoBackend() if mode == "video" else MockBackend()
    records = evaluate_video(backend, sequences, CAMUS, mode)
    assert len(records) == 2 * N_FRAMES
    names = ["lv_endo", "lv_myo", "left_atrium", "lv_epi"]
    for name in names:
        assert all(0.0 < r[f"dice_{name}"] <= 1.0 for r in records)
    first, last = records[0], records[N_FRAMES - 1]
    assert first["is_ed"] and not first["is_es"]
    assert last["is_es"] and not last["is_ed"]
    # filling the box is nearly right for blobs but poor for the wall
    assert records[0]["dice_lv_epi"] > records[0]["dice_lv_myo"]

    summary = summarize(records, CAMUS)
    assert set(summary) >= {"dice_mean", "dice_lv_epi", "dice_lv_epi_median", "ed_es"}
    assert summary["ed_es"]["n_frames"] == 4
    # dice_mean stays over the three base structures, the derived one is separate
    base = [summary[f"dice_{n}"] for n in names[:3]]
    assert summary["dice_mean"] == pytest.approx(np.mean(base))
