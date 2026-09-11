"""Registry contracts: datasets and promptable models are plug-and-play."""

import pytest

from medseg_label_efficiency.data.registry import DATASETS, INTERFACE, get_dataset_module
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
