"""Kaggle kernel: train frozen-encoder segmentation heads and evaluate them.

Submitted with:  kaggle kernels push -p kaggle/head/ --accelerator NvidiaTeslaT4

For each config: head training (frozen encoder, cached embeddings) then the
full native-protocol test evaluation. DINOv2 downloads itself via torch.hub;
DINOv3's gated weights, when used, must be attached as a Kaggle dataset and
are copied to checkpoints/ before training.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/ahmadM9/medseg-label-efficiency.git"
REPO_DIR = Path("/tmp/repo")
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

# edit before pushing a different run
CONFIGS = ["dino2_head_p05"]


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def find_data_root() -> Path:
    for pattern in ("*", "*/*", "*/*/*", "*/*/*/*"):
        for candidate in sorted(INPUT.glob(pattern)):
            if candidate.name == "database_nifti" and candidate.is_dir():
                return candidate.parent
    raise SystemExit(f"no CAMUS mount found under {INPUT}")


# attached gated snapshots: the folder Kaggle mounts (dataset slug) -> the
# path the encoder registry expects
GATED_SNAPSHOTS = {"dinov3-vits16": "checkpoints/dinov3_vits16_hf"}


def stage_gated_weights() -> None:
    """Copy attached gated-model snapshots (e.g. DINOv3) to their registry paths."""
    for f in INPUT.glob("**/model.safetensors"):
        mount_name = next(
            (p.name for p in f.parents if p.name in GATED_SNAPSHOTS), f.parent.name
        )
        target = REPO_DIR / GATED_SNAPSHOTS.get(mount_name, f"checkpoints/{f.parent.name}")
        shutil.copytree(f.parent, target, dirs_exist_ok=True)
        print(f"staged {mount_name} -> {target.relative_to(REPO_DIR)}", flush=True)


def main() -> None:
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
    run([sys.executable, "-m", "pip", "install", "-q", REPO_DIR])
    run([sys.executable, "-m", "pip", "install", "-q", "transformers"])
    stage_gated_weights()

    import yaml

    data_root = find_data_root()
    for name in CONFIGS:
        cfg = yaml.safe_load((REPO_DIR / "configs" / f"{name}.yaml").read_text())
        cfg["data_root"] = str(data_root)
        if cfg.get("train_subset"):
            cfg["train_subset"] = str(REPO_DIR / cfg["train_subset"])
        out_dir = WORK / cfg["out_dir"]
        cfg["out_dir"] = str(out_dir)
        config_path = WORK / f"config_{name}.yaml"
        config_path.write_text(yaml.safe_dump(cfg))

        env = {**os.environ, "MLFLOW_TRACKING_URI": f"sqlite:///{WORK}/mlflow.db"}
        run(
            [sys.executable, "-m", "medseg_label_efficiency.train_head",
             "--config", config_path, "--device", "cuda"],
            cwd=REPO_DIR, env=env,
        )
        run(
            [sys.executable, "-m", "medseg_label_efficiency.eval_head",
             "--config", config_path, "--device", "cuda"],
            cwd=REPO_DIR, env=env,
        )


if __name__ == "__main__":
    main()
