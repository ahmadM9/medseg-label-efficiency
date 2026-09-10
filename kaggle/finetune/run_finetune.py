"""Kaggle kernel: fine-tune the MedSAM2 decoder at all four label budgets.

Submitted with:  kaggle kernels push -p kaggle/finetune/ --accelerator NvidiaTeslaT4

For each budget config: decoder fine-tuning (5000 steps, frozen encoder,
cached embeddings), then the full native-protocol test evaluation with the
best decoder. Checkpoints and metrics land under /kaggle/working/outputs/.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/ahmadM9/medseg-label-efficiency.git"
REPO_DIR = Path("/tmp/repo")
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

CONFIGS = ["medsam2_ft_p05", "medsam2_ft_p10", "medsam2_ft_p25", "medsam2_ft_full"]


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

    import yaml

    data_root = find_data_root()
    for name in CONFIGS:
        cfg = yaml.safe_load((REPO_DIR / "configs" / f"{name}.yaml").read_text())
        cfg["data_root"] = str(data_root)
        if cfg.get("train_subset"):
            cfg["train_subset"] = str(REPO_DIR / cfg["train_subset"])
        out_dir = WORK / cfg["out_dir"]  # absolute: checkpoints must persist
        cfg["out_dir"] = str(out_dir)
        config_path = WORK / f"config_{name}.yaml"
        config_path.write_text(yaml.safe_dump(cfg))

        # cwd=REPO_DIR because the model-checkpoint paths in eval_sam's
        # CHECKPOINTS table are repo-relative
        run(
            [sys.executable, "-m", "medseg_label_efficiency.finetune_sam",
             "--config", config_path, "--device", "cuda"],
            cwd=REPO_DIR,
            env={**os.environ, "MLFLOW_TRACKING_URI": f"sqlite:///{WORK}/mlflow.db"},
        )
        run(
            [sys.executable, "-m", "medseg_label_efficiency.eval_sam",
             "--model", "medsam2", "--prompt", "box",
             "--data-root", data_root, "--device", "cuda",
             "--decoder-weights", out_dir / "decoder_best.pt",
             "--out-dir", out_dir],
            cwd=REPO_DIR,
        )


if __name__ == "__main__":
    main()
