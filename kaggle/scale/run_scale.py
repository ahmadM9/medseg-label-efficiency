"""Kaggle kernel: zero-shot regeneration, the SAM2.1 scale rows and mask re-evaluations.

Submitted with:  kaggle kernels push -p kaggle/scale/ --accelerator NvidiaTeslaT4

Jobs, in order: the four tiny-model zero-shot rows regenerated with per-image
prompt seeds and saved masks; SAM2.1 base+ and large zero-shot (box, point);
SAM2.1-large decoder fine-tuning at 20 and 40 patients; seed-0
re-evaluations that save masks for the U-Net, MedSAM2-FT and head arms
(their checkpoints must be attached under /kaggle/input, found by the
out_dir name); the in-context rows re-run with masks, plus two extra
support-draw seeds for SegGPT at 100 patients. Edit JOBS to run a part.
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

# (kind, *args): zeroshot(model, prompt), large_ft(config), reeval_unet(config),
# reeval_ft(config), reeval_head(config), incontext(config, shots, draw_seed)
JOBS = [
    ("zeroshot", "sam2", "box"), ("zeroshot", "sam2", "point"),
    ("zeroshot", "medsam2", "box"), ("zeroshot", "medsam2", "point"),
    ("zeroshot", "sam2_bplus", "box"), ("zeroshot", "sam2_bplus", "point"),
    ("zeroshot", "sam2_large", "box"), ("zeroshot", "sam2_large", "point"),
    ("large_ft", "sam2_large_ft_p05"), ("large_ft", "sam2_large_ft_p10"),
    ("reeval_unet", "camus_unet2d_p05"), ("reeval_unet", "camus_unet2d_p10"),
    ("reeval_unet", "camus_unet2d_p25"), ("reeval_unet", "camus_unet2d"),
    ("reeval_ft", "medsam2_ft_p05"), ("reeval_ft", "medsam2_ft_p10"),
    ("reeval_ft", "medsam2_ft_p25"), ("reeval_ft", "medsam2_ft_full"),
    ("reeval_head", "dino3_head_p05"), ("reeval_head", "dino3_head_p10"),
    ("reeval_head", "dino3_head_p25"), ("reeval_head", "dino3_head_full"),
    ("reeval_head", "dino2_head_p05"), ("reeval_head", "dino2_head_p10"),
    ("reeval_head", "dino2_head_p25"), ("reeval_head", "dino2_head_full"),
    ("incontext", "universeg_p05", 0, 0), ("incontext", "universeg_p10", 0, 0),
    ("incontext", "universeg_p25", 0, 0), ("incontext", "universeg_full", 0, 0),
    ("incontext", "seggpt_p05", 0, 0), ("incontext", "seggpt_p10", 0, 0),
    ("incontext", "seggpt_p25", 0, 0), ("incontext", "seggpt_full", 0, 0),
    ("incontext", "seggpt_p25", 0, 1), ("incontext", "seggpt_p25", 0, 2),
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


def find_checkpoint(out_rel: str, filename: str) -> Path:
    # attached kernel outputs and datasets both mount under /kaggle/input;
    # the checkpoint is found by its out_dir name, wherever that sits
    matches = sorted(INPUT.glob(f"**/{Path(out_rel).name}/{filename}"))
    if not matches:
        raise SystemExit(f"no {filename} for {out_rel} under {INPUT}")
    if len(matches) > 1:
        print(f"several {filename} for {out_rel}, taking {matches[0]}", flush=True)
    return matches[0]


def write_config(name: str, data_root: Path) -> tuple[Path, dict]:
    import yaml

    cfg = yaml.safe_load((REPO_DIR / "configs" / f"{name}.yaml").read_text())
    cfg["data_root"] = str(data_root)
    if cfg.get("train_subset"):
        cfg["train_subset"] = str(REPO_DIR / cfg["train_subset"])
    cfg["out_dir"] = str(WORK / cfg["out_dir"])
    path = WORK / f"config_{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


def main() -> None:
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
    run([sys.executable, "-m", "pip", "install", "-q", REPO_DIR])
    kinds = {job[0] for job in JOBS}
    if kinds & {"zeroshot", "large_ft", "reeval_ft"}:
        run(
            [sys.executable, "-m", "pip", "install", "-q",
             "git+https://github.com/facebookresearch/sam2.git"],
            env={**os.environ, "SAM2_BUILD_CUDA": "0"},
        )
        run(["bash", REPO_DIR / "scripts" / "download_checkpoints.sh"])
    if "reeval_head" in kinds:
        run([sys.executable, "-m", "pip", "install", "-q", "transformers"])
        stage_gated_weights()
    if "incontext" in kinds:
        run([sys.executable, "-m", "pip", "install", "-q", "transformers",
             "git+https://github.com/JJGO/UniverSeg.git"])

    data_root = find_data_root()
    env = {**os.environ, "MLFLOW_TRACKING_URI": f"sqlite:///{WORK}/mlflow.db"}
    for job in JOBS:
        kind = job[0]
        print(f"job {job}", flush=True)
        if kind == "zeroshot":
            _, model, prompt = job
            run([sys.executable, "-m", "medseg_label_efficiency.eval_sam",
                 "--model", model, "--prompt", prompt, "--data-root", data_root,
                 "--device", "cuda", "--out-dir", WORK / "outputs" / f"{model}_{prompt}",
                 "--save-masks"], cwd=REPO_DIR)
        elif kind == "large_ft":
            config_path, cfg = write_config(job[1], data_root)
            run([sys.executable, "-m", "medseg_label_efficiency.finetune_sam", "--config",
                 config_path, "--device", "cuda"], cwd=REPO_DIR, env=env)
            run([sys.executable, "-m", "medseg_label_efficiency.eval_sam",
                 "--model", cfg["model"], "--prompt", "box", "--data-root", data_root,
                 "--device", "cuda", "--decoder-weights", Path(cfg["out_dir"]) / "decoder_best.pt",
                 "--out-dir", cfg["out_dir"], "--save-masks"], cwd=REPO_DIR)
        elif kind == "reeval_unet":
            config_path, cfg = write_config(job[1], data_root)
            run([sys.executable, "-m", "medseg_label_efficiency.evaluate", "--config",
                 config_path, "--device", "cuda", "--save-masks",
                 "--checkpoint", find_checkpoint(cfg["out_dir"], "best.pt")], cwd=REPO_DIR)
        elif kind == "reeval_ft":
            config_path, cfg = write_config(job[1], data_root)
            run([sys.executable, "-m", "medseg_label_efficiency.eval_sam",
                 "--model", cfg.get("model", "medsam2"), "--prompt", "box",
                 "--data-root", data_root, "--device", "cuda",
                 "--decoder-weights", find_checkpoint(cfg["out_dir"], "decoder_best.pt"),
                 "--out-dir", cfg["out_dir"], "--save-masks"], cwd=REPO_DIR)
        elif kind == "reeval_head":
            config_path, cfg = write_config(job[1], data_root)
            run([sys.executable, "-m", "medseg_label_efficiency.eval_head", "--config",
                 config_path, "--device", "cuda", "--save-masks",
                 "--checkpoint", find_checkpoint(cfg["out_dir"], "head_best.pt")],
                cwd=REPO_DIR)
        elif kind == "incontext":
            _, name, shots, draw_seed = job
            config_path, cfg = write_config(name, data_root)
            cmd = [sys.executable, "-m", "medseg_label_efficiency.eval_incontext", "--config",
                   config_path, "--device", "cuda", "--save-masks"]
            if shots:
                cmd += ["--shots", shots]
            if draw_seed:
                cmd += ["--draw-seed", draw_seed]
            run(cmd, cwd=REPO_DIR)
        else:
            raise SystemExit(f"unknown job kind {kind!r}")


if __name__ == "__main__":
    main()
