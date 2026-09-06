"""Tests against the real CAMUS copy — skipped automatically if not linked."""

import numpy as np
import pytest

from medseg_label_efficiency.data.camus import build_samples, get_dataset, read_split
from medseg_label_efficiency.data.verify import verify

pytestmark = pytest.mark.requires_data


def test_verify_passes(data_root):
    assert verify(data_root) == []


def test_official_split_sizes(data_root):
    assert len(read_split(data_root, "train")) == 400
    assert len(read_split(data_root, "val")) == 50
    assert len(read_split(data_root, "test")) == 50


def test_no_patient_leakage(data_root):
    splits = [set(read_split(data_root, s)) for s in ("train", "val", "test")]
    assert not (splits[0] & splits[1])
    assert not (splits[0] & splits[2])
    assert not (splits[1] & splits[2])


def test_sample_counts(data_root):
    # 4 labeled images per patient: {2CH,4CH} x {ED,ES}
    assert len(build_samples(data_root, "train")) == 1600
    assert len(build_samples(data_root, "val")) == 200
    assert len(build_samples(data_root, "test")) == 200


def test_sample_metadata(data_root):
    sample = build_samples(data_root, "val")[0]
    assert sample["view"] in ("2CH", "4CH")
    assert sample["phase"] in ("ED", "ES")
    assert sample["quality"] in ("Good", "Medium", "Poor")


@pytest.mark.parametrize("index", [0, 100, 199])
def test_dataset_loads_real_samples(data_root, index):
    ds = get_dataset(data_root, "val")
    out = ds[index]
    assert out["image"].shape == (1, 256, 256)
    assert out["label"].shape == (1, 256, 256)
    labels = set(np.unique(out["label"].numpy()).tolist())
    assert labels <= {0.0, 1.0, 2.0, 3.0}
    assert len(labels) > 1  # a labeled frame should contain foreground
