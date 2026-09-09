"""Kaggle kernel that trains the CAMUS baseline on a Kaggle GPU.

Submitted with the Kaggle CLI from the repo root:

    kaggle kernels push -p kaggle/

The kernel machine starts empty, so this script clones the public repo,
installs it, points it at the mounted CAMUS dataset and trains. Everything
under /kaggle/working survives as the kernel's output, which is how
checkpoints, MLflow data and metrics get back to us. If a previous run's
output is attached as an input, training resumes from its latest checkpoint
instead of starting over (Kaggle sessions are capped at 12 hours).
"""

import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_URL = "https://github.com/ahmadM9/medseg-label-efficiency.git"
REPO_DIR = Path("/tmp/repo")
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def find_data_root() -> Path:
    for candidate in sorted(INPUT.glob("*")):
        if (candidate / "database_nifti").is_dir():
            return candidate
    raise SystemExit(f"no CAMUS mount with database_nifti found under {INPUT}")


def find_resume_checkpoint() -> Path | None:
    for prev in sorted(INPUT.glob("*/outputs/camus_unet2d/latest.pt")):
        return prev.parent
    return None


def main() -> None:
    run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
    run([sys.executable, "-m", "pip", "install", "-q", REPO_DIR])

    data_root = find_data_root()
    run([sys.executable, "-m", "medseg_label_efficiency.data.verify", "--data-root", data_root])

    cfg = yaml.safe_load((REPO_DIR / "configs" / "camus_unet2d.yaml").read_text())
    cfg["data_root"] = str(data_root)
    cfg["out_dir"] = "outputs/camus_unet2d"
    config_path = WORK / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    train_cmd = [sys.executable, "-m", "medseg_label_efficiency.train", "--config", config_path]
    prev = find_resume_checkpoint()
    if prev is not None:
        print(f"found previous checkpoints at {prev}, resuming", flush=True)
        dest = WORK / "outputs" / "camus_unet2d"
        dest.mkdir(parents=True, exist_ok=True)
        for f in prev.glob("*.pt"):
            shutil.copy2(f, dest / f.name)
        train_cmd.append("--resume")

    run(train_cmd, cwd=WORK)


if __name__ == "__main__":
    main()
