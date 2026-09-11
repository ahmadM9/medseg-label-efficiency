"""Tests against the real CAMUS copy — skipped automatically if not linked."""

from pathlib import Path

import numpy as np
import pytest

from medseg_label_efficiency.data.camus import build_samples, get_dataset, read_split, verify

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


def test_patient_subset_filtering(data_root):
    subset = read_split(data_root, "train")[:3]
    ds = get_dataset(data_root, "train", patients=subset)
    assert len(ds) == 4 * len(subset)  # 2 views x 2 phases per patient
    assert {s["patient"] for s in ds.data} == set(subset)


def test_committed_subsets_nest_and_come_from_train_split(data_root):
    subset_dir = Path(__file__).parents[1] / "configs" / "subsets"
    train = set(read_split(data_root, "train"))
    lists = {
        name: set((subset_dir / f"train_{name}.txt").read_text().split())
        for name in ("p05", "p10", "p25")
    }
    assert len(lists["p05"]) == 20 and len(lists["p10"]) == 40 and len(lists["p25"]) == 100
    assert lists["p05"] <= lists["p10"] <= lists["p25"] <= train


@pytest.mark.parametrize("index", [0, 100, 199])
def test_dataset_loads_real_samples(data_root, index):
    ds = get_dataset(data_root, "val")
    out = ds[index]
    assert out["image"].shape == (1, 256, 256)
    assert out["label"].shape == (1, 256, 256)
    labels = set(np.unique(out["label"].numpy()).tolist())
    assert labels <= {0.0, 1.0, 2.0, 3.0}
    assert len(labels) > 1  # a labeled frame should contain foreground
