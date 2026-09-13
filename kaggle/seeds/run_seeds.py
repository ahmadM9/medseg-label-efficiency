"""Kaggle kernel: re-run the seeded arms with a new subset draw and training seed.

Submitted with:  kaggle kernels push -p kaggle/seeds/ --accelerator NvidiaTeslaT4

Each job is (arm, config name, seed). Seed 0 is the committed config as it
stands (seed 42, the train_p* lists); a seed s > 0 injects seed s, the
subset list train_p*_s<s>.txt when the budget is below the full set, and
writes to <out_dir>_s<s>. Every evaluation saves its masks. The arms are
unet (train + evaluate), ft (finetune_sam + eval_sam), head (train_head +
eval_head, DINOv3 weights attached as a dataset) and universeg
(eval_incontext, no training, so only the subset changes).
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

# edit before pushing; the dry run is the single U-Net job below
JOBS = [
    ("unet", "camus_unet2d_p05", 1),
]

GATED_SNAPSHOTS = {"dinov3-vits16": "checkpoints/dinov3_vits16_hf"}


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def find_data_root() -> Path:
    for pattern in ("*", "*/*", "*/*/*", "*/*/*/*"):
        for candidate in sorted(INPUT.glob(pattern)):
            if candidate.name == "database_nifti" and candidate.is_dir():
                return candidate.parent
    raise SystemExit(f"no CAMUS mount found under {INPUT}")


def stage_gated_weights() -> None:
    for f in INPUT.glob("**/model.safetensors"):
        mount_name = next(
            (p.name for p in f.parents if p.name in GATED_SNAPSHOTS), f.parent.name
        )
        target = REPO_DIR / GATED_SNAPSHOTS.get(mount_name, f"checkpoints/{f.parent.name}")
        shutil.copytree(f.parent, target, dirs_exist_ok=True)
        print(f"staged {mount_name} -> {target.relative_to(REPO_DIR)}", flush=True)


def write_config(name: str, seed: int, data_root: Path) -> tuple[Path, dict]:
    import yaml

    cfg = yaml.safe_load((REPO_DIR / "configs" / f"{name}.yaml").read_text())
    cfg["data_root"] = str(data_root)
    if seed > 0:
        cfg["seed"] = seed
        if cfg.get("train_subset"):
            # the seed names both the subset draw and the training seed
            cfg["train_subset"] = cfg["train_subset"].replace(".txt", f"_s{seed}.txt")
        cfg["out_dir"] = f"{cfg['out_dir']}_s{seed}"
    if cfg.get("train_subset"):
        cfg["train_subset"] = str(REPO_DIR / cfg["train_subset"])
    cfg["out_dir"] = str(WORK / cfg["out_dir"])  # absolute: outputs must persist
    path = WORK / f"config_{name}_s{seed}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


def find_resume_checkpoint(out_dir: Path, filename: str) -> Path | None:
    for prev in sorted(INPUT.glob(f"**/{out_dir.name}/{filename}")):
        return prev.parent
    return None


def main() -> None:
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
    run([sys.executable, "-m", "pip", "install", "-q", REPO_DIR])
    arms = {arm for arm, _, _ in JOBS}
    if "ft" in arms:
        run(
            [sys.executable, "-m", "pip", "install", "-q",
             "git+https://github.com/facebookresearch/sam2.git"],
            env={**os.environ, "SAM2_BUILD_CUDA": "0"},
        )
        run(["bash", REPO_DIR / "scripts" / "download_checkpoints.sh"])
    if "head" in arms:
        run([sys.executable, "-m", "pip", "install", "-q", "transformers"])
        stage_gated_weights()
    if "universeg" in arms:
        run([sys.executable, "-m", "pip", "install", "-q",
             "git+https://github.com/JJGO/UniverSeg.git"])

    data_root = find_data_root()
    env = {**os.environ, "MLFLOW_TRACKING_URI": f"sqlite:///{WORK}/mlflow.db"}
    for arm, name, seed in JOBS:
        config_path, cfg = write_config(name, seed, data_root)
        out_dir = Path(cfg["out_dir"])
        print(f"job {arm} {name} seed {seed} -> {out_dir}", flush=True)
        if arm == "unet":
            cmd = [sys.executable, "-m", "medseg_label_efficiency.train", "--config", config_path]
            prev = find_resume_checkpoint(out_dir, "latest.pt")
            if prev is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                for f in prev.glob("*.pt"):
                    shutil.copy2(f, out_dir / f.name)
                cmd.append("--resume")
            run(cmd, cwd=REPO_DIR, env=env)
            run([sys.executable, "-m", "medseg_label_efficiency.evaluate", "--config",
                 config_path, "--device", "cuda", "--save-masks"], cwd=REPO_DIR, env=env)
        elif arm == "ft":
            run([sys.executable, "-m", "medseg_label_efficiency.finetune_sam", "--config",
                 config_path, "--device", "cuda"], cwd=REPO_DIR, env=env)
            run([sys.executable, "-m", "medseg_label_efficiency.eval_sam",
                 "--model", cfg.get("model", "medsam2"), "--prompt", "box",
                 "--data-root", data_root, "--device", "cuda",
                 "--decoder-weights", out_dir / "decoder_best.pt",
                 "--out-dir", out_dir, "--save-masks"], cwd=REPO_DIR)
        elif arm == "head":
            run([sys.executable, "-m", "medseg_label_efficiency.train_head", "--config",
                 config_path, "--device", "cuda"], cwd=REPO_DIR, env=env)
            run([sys.executable, "-m", "medseg_label_efficiency.eval_head", "--config",
                 config_path, "--device", "cuda", "--save-masks"], cwd=REPO_DIR, env=env)
        elif arm == "universeg":
            run([sys.executable, "-m", "medseg_label_efficiency.eval_incontext", "--config",
                 config_path, "--device", "cuda", "--save-masks"], cwd=REPO_DIR)
        else:
            raise SystemExit(f"unknown arm {arm!r}")


if __name__ == "__main__":
    main()
