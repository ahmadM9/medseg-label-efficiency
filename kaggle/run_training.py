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
import time
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
    # newer Kaggle runtimes mount datasets at /kaggle/input/datasets/<user>/<slug>,
    # older ones at /kaggle/input/<slug>, so search a few levels; the mount can
    # also lag behind kernel start, hence the retries
    patterns = ("*", "*/*", "*/*/*", "*/*/*/*")
    for attempt in range(6):
        for pattern in patterns:
            for candidate in sorted(INPUT.glob(pattern)):
                if candidate.name == "database_nifti" and candidate.is_dir():
                    return candidate.parent
        print(f"attempt {attempt + 1}: no database_nifti yet under {INPUT}", flush=True)
        time.sleep(20)
    listing = "\n".join(str(p) for pat in patterns for p in sorted(INPUT.glob(pat))[:15])
    raise SystemExit(f"no CAMUS mount found under {INPUT}; contents:\n{listing or '(empty)'}")


def find_resume_checkpoint() -> Path | None:
    for prev in sorted(INPUT.glob("**/outputs/camus_unet2d/latest.pt")):
        return prev.parent
    return None


def main() -> None:
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
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
