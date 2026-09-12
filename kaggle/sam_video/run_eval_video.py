"""Kaggle kernel for the video-mode versus still-mode cross-check on CAMUS.

Submitted with:  kaggle kernels push -p kaggle/sam_video/ --accelerator NvidiaTeslaT4

Clones the repo, installs it plus the sam2 library, fetches both checkpoints,
then runs {medsam2, sam2} x {video, still} over the 100 test half-cycle
sequences. Results land in /kaggle/working/outputs/<model>_<mode>_video/.
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
    for model in ("medsam2", "sam2"):
        for mode in ("video", "still"):
            run(
                [sys.executable, "-m", "medseg_label_efficiency.eval_video",
                 "--model", model, "--mode", mode,
                 "--data-root", data_root, "--device", "cuda",
                 "--out-dir", WORK / "outputs" / f"{model}_{mode}_video"],
                cwd=REPO_DIR,
            )


if __name__ == "__main__":
    main()
