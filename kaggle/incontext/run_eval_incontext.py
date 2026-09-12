"""Kaggle kernel that runs the in-context arms on the full CAMUS test split.

Submitted with:  kaggle kernels push -p kaggle/incontext/ --accelerator NvidiaTeslaT4

Clones the repo, installs it plus the universeg package and transformers
(SegGPT weights come from the HuggingFace hub, public), then evaluates every
job in JOBS on the GPU. Results land in /kaggle/working/outputs/<model>_<budget>/
with a _k<shots> suffix for the supplementary support sizes.
"""

import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/ahmadM9/medseg-label-efficiency.git"
REPO_DIR = Path("/tmp/repo")
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

# (config name, --shots override or None, --limit or 0). universeg main rows
# use the config's 32, the matched row 8, and 64 where the pool allows it
JOBS = [
    ("universeg_p05", None, 0),
    ("universeg_p05", 8, 0),
    ("universeg_p10", None, 0),
    ("universeg_p10", 8, 0),
    ("universeg_p10", 64, 0),
    ("universeg_p25", None, 0),
    ("universeg_p25", 8, 0),
    ("universeg_p25", 64, 0),
    ("universeg_full", None, 0),
    ("universeg_full", 8, 0),
    ("universeg_full", 64, 0),
    ("seggpt_p05", None, 0),
    ("seggpt_p10", None, 0),
    ("seggpt_p25", None, 0),
    ("seggpt_full", None, 0),
]


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
    run([sys.executable, "-m", "pip", "install", "-q", "transformers",
         "git+https://github.com/JJGO/UniverSeg.git"])

    import yaml

    data_root = find_data_root()
    for name, shots, limit in JOBS:
        cfg = yaml.safe_load((REPO_DIR / "configs" / f"{name}.yaml").read_text())
        cfg["data_root"] = str(data_root)
        if cfg.get("train_subset"):
            cfg["train_subset"] = str(REPO_DIR / cfg["train_subset"])
        cfg["out_dir"] = str(WORK / cfg["out_dir"])
        config_path = WORK / f"config_{name}.yaml"
        config_path.write_text(yaml.safe_dump(cfg))

        cmd = [sys.executable, "-m", "medseg_label_efficiency.eval_incontext",
               "--config", config_path, "--device", "cuda"]
        if shots:
            cmd += ["--shots", shots]
        if limit:
            cmd += ["--limit", limit]
        run(cmd, cwd=REPO_DIR)


if __name__ == "__main__":
    main()
