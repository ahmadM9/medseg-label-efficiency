"""Registry contracts: datasets, promptable and in-context models are plug-and-play."""

import pytest

from medseg_label_efficiency.data.registry import DATASETS, INTERFACE, get_dataset_module
from medseg_label_efficiency.incontext import IN_CONTEXT_MODELS
from medseg_label_efficiency.promptable import PROMPTABLE_MODELS


def test_lookup_returns_registered_module():
    assert get_dataset_module("camus") is DATASETS["camus"]


def test_unknown_dataset_error_lists_known_names():
    with pytest.raises(KeyError, match="camus"):
        get_dataset_module("no_such_dataset")


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_registered_datasets_satisfy_interface(name):
    module = get_dataset_module(name)
    for attr in INTERFACE:
        assert hasattr(module, attr), f"{name} is missing {attr}"
    assert set(module.STRUCTURE_TITLES) == set(module.STRUCTURES.values())


def test_promptable_registry_entries_are_complete():
    for name, entry in PROMPTABLE_MODELS.items():
        assert {"checkpoint", "config", "vendored_config"} <= set(entry), name


def test_resolution_control_shares_weights_and_config():
    # sam2_512 must be the stock tiny weights in MedSAM2's 512 config, or the
    # pair (sam2_ft, sam2_512_ft) stops isolating the input size
    assert PROMPTABLE_MODELS["sam2_512"]["checkpoint"] == PROMPTABLE_MODELS["sam2"]["checkpoint"]
    assert PROMPTABLE_MODELS["sam2_512"]["config"] == PROMPTABLE_MODELS["medsam2"]["config"]
    assert PROMPTABLE_MODELS["sam2_512"]["vendored_config"] == (
        PROMPTABLE_MODELS["medsam2"]["vendored_config"]
    )


def test_incontext_registry_entries_are_complete():
    for name, entry in IN_CONTEXT_MODELS.items():
        assert {"kind", "source", "input_size", "training_data"} <= set(entry), name
        assert entry["kind"] in ("universeg", "transformers"), name
