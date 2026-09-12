import nibabel as nib
import numpy as np
import pytest

from medseg_label_efficiency.data.registry import get_dataset_module
from medseg_label_efficiency.eval_incontext import evaluate_model
from medseg_label_efficiency.eval_sam import summarize
from medseg_label_efficiency.incontext import (
    draw_supports,
    filter_pool,
    load_pool,
    to_square,
)
from medseg_label_efficiency.prompts import sample_rng

CAMUS = get_dataset_module("camus")
SIZE = 32


class MockBackend:
    # the prediction is the average of the support masks: sensitive to which
    # supports were drawn, which is what the harness tests care about
    def predict_draw(self, query, support_images, support_masks):
        assert query.shape == (SIZE, SIZE) and query.dtype == np.uint8
        assert support_images.shape[1:] == (SIZE, SIZE)
        return support_masks.mean(axis=0)

    @staticmethod
    def combine(draws):
        return np.mean(draws, axis=0) > 0.5


def _write(tmp_path, name, label, shift=0):
    image = np.random.default_rng(0).random((60, 50)).astype(np.float32) * 200
    affine = np.diag([0.5, 0.4, 1.0, 1.0])
    nib.save(nib.Nifti1Image(image, affine), tmp_path / f"{name}.nii.gz")
    nib.save(nib.Nifti1Image(label, affine), tmp_path / f"{name}_gt.nii.gz")
    return {"image": str(tmp_path / f"{name}.nii.gz"), "label": str(tmp_path / f"{name}_gt.nii.gz")}


def _label(shift=0):
    label = np.zeros((60, 50), dtype=np.uint8)
    label[15 + shift : 30 + shift, 15:30] = 1
    label[10 + shift : 15 + shift, 15:30] = 2
    label[35 + shift : 48 + shift, 18:33] = 3
    return label


@pytest.fixture
def pool(tmp_path):
    samples = []
    for p, shift in (("patient0001", 0), ("patient0002", 1), ("patient0003", 2)):
        for view in ("2CH", "4CH"):
            for phase in ("ED", "ES"):
                s = _write(tmp_path, f"{p}_{view}_{phase}", _label(shift))
                s.update(patient=p, view=view, phase=phase, quality="Good")
                samples.append(s)
    return samples


@pytest.fixture
def query(tmp_path):
    s = _write(tmp_path, "patient0101_2CH_ED", _label(0))
    s.update(patient="patient0101", view="2CH", phase="ED", quality="Medium")
    return s


def test_pool_filter_matches_view_and_drops_query_patient(pool, query):
    idx = filter_pool(pool, query, ["view"])
    assert len(idx) == 6
    assert all(pool[i]["view"] == "2CH" for i in idx)
    same_patient = dict(query, patient="patient0002")
    idx = filter_pool(pool, same_patient, ["view"])
    assert len(idx) == 4 and all(pool[i]["patient"] != "patient0002" for i in idx)
    assert len(filter_pool(pool, query, [])) == 12


def test_draws_deterministic_distinct_and_bounded():
    candidates = np.arange(10)
    a = draw_supports(candidates, 6, sample_rng("patient0001", "2CH", "ED", "Good", 1))
    b = draw_supports(candidates, 6, sample_rng("patient0001", "2CH", "ED", "Good", 1))
    c = draw_supports(candidates, 6, sample_rng("patient0001", "2CH", "ED", "Good", 2))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert len(set(a.tolist())) == 6
    with pytest.raises(ValueError, match="12 supports requested but only 10"):
        draw_supports(candidates, 12, sample_rng("x"))


def test_to_square_keeps_label_values_and_image_range():
    label = _label()
    small = to_square(label, SIZE, "nearest")
    assert small.shape == (SIZE, SIZE) and set(np.unique(small)) <= {0, 1, 2, 3}
    image = np.random.default_rng(1).integers(0, 256, (60, 50)).astype(np.uint8)
    assert to_square(image, SIZE, "bilinear").max() <= 255


def test_harness_records_and_overlap(pool, query):
    images, labels = load_pool(pool, SIZE)
    assert images.shape == (12, SIZE, SIZE) and labels.dtype == np.uint8
    records = evaluate_model(
        MockBackend(), [query], pool, images, labels, CAMUS, shots=4, draws=3, match_keys=["view"]
    )
    row = records[0]
    assert row["patient"] == "patient0101" and row["quality"] == "Medium"
    for name in CAMUS.STRUCTURES.values():
        assert 0.3 < row[f"dice_{name}"] <= 1.0
        assert np.isfinite(row[f"hd95_{name}"])
    # mock predictions are averaged copies of disjoint labels, so no overlap
    assert row["overlap_frac"] == 0.0
    summary = summarize(records, CAMUS)
    assert set(summary) >= {"dice_mean", "hd95_mean", "by_quality"}


def test_overlap_is_measured(pool, query):
    class Everything(MockBackend):
        # every structure claims the whole image
        @staticmethod
        def combine(draws):
            return np.ones_like(draws[0], dtype=bool)

    images, labels = load_pool(pool, SIZE)
    records = evaluate_model(
        Everything(), [query], pool, images, labels, CAMUS, shots=2, draws=1, match_keys=["view"]
    )
    assert records[0]["overlap_frac"] == 1.0
