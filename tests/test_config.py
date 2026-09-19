"""A config naming a missing subset list fails at load time."""

import pytest

from medseg_label_efficiency.config import load_config


def test_missing_subset_list_fails_at_load(tmp_path):
    subset = tmp_path / "train_p05.txt"
    subset.write_text("patient0001\n")
    good = tmp_path / "good.yaml"
    good.write_text(f"dataset: camus\ntrain_subset: {subset}\n")
    assert load_config(good)["train_subset"] == str(subset)
    full = tmp_path / "full.yaml"
    full.write_text("dataset: camus\n")
    assert load_config(full).get("train_subset") is None
    bad = tmp_path / "bad.yaml"
    bad.write_text("dataset: camus\ntrain_subset: None\n")
    with pytest.raises(FileNotFoundError, match="train_subset"):
        load_config(bad)
