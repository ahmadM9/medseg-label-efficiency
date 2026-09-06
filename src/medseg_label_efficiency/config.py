from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file into a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)
