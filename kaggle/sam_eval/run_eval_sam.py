"""Kaggle kernel that runs the zero-shot arms on the full CAMUS test split.

Submitted with:  kaggle kernels push -p kaggle/sam_eval/ --accelerator NvidiaTeslaT4

Clones the repo, installs it plus the sam2 library, fetches both checkpoints,
then evaluates all four configurations ({sam2, medsam2} x {box, point}) on
the GPU. Results land in /kaggle/working/outputs/<model>_<prompt>/.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/ahmadM9/medseg-label-efficiency.git"
REPO_DIR = Path("/tmp/repo")
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def find_data_root() -> Path:
    for pattern in ("*", "*/*", "*/*/*", "*/*/*/*"):
        for candidate in sorted(INPUT.glob(pattern)):
            if candidate.name == "database_nifti" and candidate.is_dir():
                return candidate.parent
    raise SystemExit(f"no CAMUS mount found under {INPUT}")


def main() -> None:
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
    run([sys.executable, "-m", "pip", "install", "-q", REPO_DIR])
    run(
        [sys.executable, "-m", "pip", "install", "-q",
         "git+https://github.com/facebookresearch/sam2.git"],
        env={**os.environ, "SAM2_BUILD_CUDA": "0"},
    )
    run(["bash", REPO_DIR / "scripts" / "download_checkpoints.sh"])

    data_root = find_data_root()
    for model in ("sam2", "medsam2"):
        for prompt in ("box", "point"):
            run(
                [sys.executable, "-m", "medseg_label_efficiency.eval_sam",
                 "--model", model, "--prompt", prompt,
                 "--data-root", data_root, "--device", "cuda",
                 "--out-dir", WORK / "outputs" / f"{model}_{prompt}"],
                cwd=REPO_DIR,
            )


if __name__ == "__main__":
    main()
