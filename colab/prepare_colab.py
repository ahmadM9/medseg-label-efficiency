"""Colab session setup: pull the private Kaggle datasets and stage checkpoints from Drive.

    python colab/prepare_colab.py [--input /content/input] \
        [--drive /content/drive/MyDrive/medseg-label-efficiency]

Reads KAGGLE_USERNAME and KAGGLE_KEY from the environment, or from Colab
Secrets when run inside the notebook kernel itself (a %%bash cell is a
subprocess and cannot see the secret store; the launcher notebook exports
them first), downloads CAMUS and the gated DINOv3 snapshot
into <input>/<slug>/ on the local disk, and copies the stripped checkpoints
from <drive>/checkpoints/outputs into <input>/checkpoints/outputs. Every step
is skipped when its target already exists, so a rerun after a disconnect costs
nothing. Data lives on the local disk because Drive is a network mount and
2000 small NIfTI files read slowly through it; outputs go to Drive instead.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

DATASETS = ("ahmadm10/camus-echo", "ahmadm10/dinov3-vits16")
SECRETS = ("KAGGLE_USERNAME", "KAGGLE_KEY")


def kaggle_credentials() -> dict[str, str]:
    env = {}
    for name in SECRETS:
        value = os.environ.get(name)
        if not value:
            try:
                from google.colab import userdata

                value = userdata.get(name)
            except Exception:  # outside Colab there is no userdata module
                value = None
        if not value:
            raise SystemExit(f"{name} is neither a Colab secret nor set in the environment")
        env[name] = value
    return env


def pull_dataset(slug: str, input_root: Path, env: dict[str, str]) -> Path:
    target = input_root / slug.split("/")[1]
    if target.exists() and any(target.iterdir()):
        print(f"{slug}: already under {target}", flush=True)
        return target
    target.mkdir(parents=True, exist_ok=True)
    # the API delivers the files as uploaded, so CAMUS keeps its .nii.gz names
    # (the Kaggle mount gunzips them; the loader accepts both spellings)
    # the console script, not python -m: the package has no __main__ on Colab
    cmd = [shutil.which("kaggle") or "kaggle", "datasets", "download", "-d", slug,
           "-p", str(target), "--unzip"]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, **env})
    return target


def stage_checkpoints(drive_root: Path, input_root: Path) -> Path | None:
    src = drive_root / "checkpoints" / "outputs"
    dst = input_root / "checkpoints" / "outputs"
    if not src.exists():
        print(f"no checkpoints under {src}; re-evaluation jobs will not find their weights")
        return None
    if dst.exists():
        print(f"checkpoints already staged under {dst}")
        return dst
    shutil.copytree(src, dst)
    n = sum(1 for _ in dst.rglob("*.pt"))
    print(f"staged {n} checkpoint files under {dst}")
    return dst


def find_database_nifti(input_root: Path) -> Path:
    for pattern in ("*", "*/*", "*/*/*", "*/*/*/*"):
        for candidate in sorted(input_root.glob(pattern)):
            if candidate.name == "database_nifti" and candidate.is_dir():
                return candidate
    raise SystemExit(f"no database_nifti under {input_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/content/input")
    parser.add_argument("--drive", default="/content/drive/MyDrive/medseg-label-efficiency")
    args = parser.parse_args()
    input_root = Path(args.input)
    env = kaggle_credentials()
    for slug in DATASETS:
        pull_dataset(slug, input_root, env)
    stage_checkpoints(Path(args.drive), input_root)
    print(f"CAMUS at {find_database_nifti(input_root).parent}")


if __name__ == "__main__":
    main()
