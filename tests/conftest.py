import sys
from pathlib import Path

import pytest

DATA_ROOT = Path(__file__).parents[1] / "data" / "camus"

# the analysis scripts are plain files, not a package; tests import them
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))


def pytest_collection_modifyitems(config, items):
    if (DATA_ROOT / "database_nifti").exists():
        return
    skip = pytest.mark.skip(reason="CAMUS data not linked under data/camus")
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def data_root() -> Path:
    return DATA_ROOT
