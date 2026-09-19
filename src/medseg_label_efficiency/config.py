from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file into a plain dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # a subset list that does not exist must fail here, not after the model
    # is built; a bare "None" in the file arrives as the string "None"
    subset = cfg.get("train_subset")
    if subset is not None and not Path(subset).is_file():
        raise FileNotFoundError(f"{path}: train_subset {subset!r} is not a file")
    return cfg
