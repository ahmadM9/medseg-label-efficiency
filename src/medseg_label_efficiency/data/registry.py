"""Dataset registry — the plug-and-play point for datasets.

A dataset is one module exposing the same small interface; everything else
asks this registry by name (usually the config's ``dataset:`` key) and never
imports a concrete dataset directly. Adding a dataset = one new module here,
one entry in DATASETS, zero edits elsewhere.

Required module attributes:
    build_samples(data_root, split) -> list[dict]   one dict per labeled image
    get_transforms(train, image_size) -> Compose
    get_dataset(data_root, split, ...) -> Dataset
    verify(data_root) -> list[str]                  problems; empty = OK
    SPLITS: tuple[str, ...]
    STRUCTURES: dict[int, str]                      label id -> slug
    STRUCTURE_TITLES: dict[str, str]                slug -> display name
    META_KEYS: tuple[str, ...]                      sample keys carried into records
"""

from medseg_label_efficiency.data import camus

DATASETS = {"camus": camus}

INTERFACE = (
    "build_samples", "get_transforms", "get_dataset", "verify",
    "SPLITS", "STRUCTURES", "STRUCTURE_TITLES", "META_KEYS",
)


def get_dataset_module(name: str):
    if name not in DATASETS:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"unknown dataset {name!r}; registered datasets: {known}")
    return DATASETS[name]
