"""nnU-Net row: convert, preprocess, train one fold per budget, predict, export.

Colab (see colab/launcher.ipynb, T4):
    LAUNCH_INPUT=/content/input LAUNCH_WORK=<drive work dir> LAUNCH_REPO=/content/repo \\
    python colab/run_nnunet.py --budgets p05
Laptop smoke (CPU, two patients, one epoch, batch size 2):
    LAUNCH_INPUT=data LAUNCH_WORK=/tmp/nnunet-smoke LAUNCH_LOCAL=/tmp/nnunet-smoke \\
    LAUNCH_REPO=. python colab/run_nnunet.py --smoke --skip-install

Raw and preprocessed data live on the local disk (LAUNCH_LOCAL, thousands of
small arrays), nnU-Net's results folder and the exported rows on LAUNCH_WORK
(Drive). Steps, each skipped when its output already exists so a rerun after
a disconnect continues where it stopped: prepare (scripts/prepare_nnunet.py),
plan and preprocess with dataset verification, copy splits_final.json next to
the preprocessed data, then per budget: train fold (--c continues from the
last checkpoint), predict imagesTs with nnU-Net's defaults (mirroring on) and
once more with --disable_tta, determine the post-processing on the fold's
validation predictions, apply it to both prediction sets, export both through
scripts/export_nnunet.py into outputs/nnunet_<budget> and
outputs/nnunet_nomirror_<budget>. The 100-epoch trainer is nnU-Net's own
variant, chosen to match the CAMUS group's nnU-Net comparison.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(os.environ.get("LAUNCH_REPO", "/content/repo"))
WORK = Path(os.environ.get("LAUNCH_WORK", "/content/drive/MyDrive/medseg-label-efficiency/work"))
INPUT = Path(os.environ.get("LAUNCH_INPUT", "/content/input"))
LOCAL = Path(os.environ.get("LAUNCH_LOCAL", "/content"))

DATASET_ID = "501"
DATASET_NAME = "Dataset501_CAMUS"
CONFIG = "2d"
TRAINER = "nnUNetTrainer_100epochs"
SMOKE_TRAINER = "nnUNetTrainer_1epoch"
# mirrors BUDGET_OF_FOLD in scripts/prepare_nnunet.py, which writes the split file
FOLD_OF_BUDGET = {"p05": 0, "p10": 1, "p25": 2, "full": 3}


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def tool(name: str) -> str:
    # pip puts console scripts on the PATH (/usr/local/bin on Colab), which is
    # not the interpreter's folder there; the venv on the laptop has both in one
    found = shutil.which(name) or shutil.which(name, path=str(Path(sys.executable).parent))
    if not found:
        raise SystemExit(f"{name} not found; is nnunetv2 installed?")
    return found


def find_data_root() -> Path:
    for pattern in ("*", "*/*", "*/*/*", "*/*/*/*"):
        for candidate in sorted(INPUT.glob(pattern)):
            if candidate.name == "database_nifti" and candidate.is_dir():
                return candidate.parent
    raise SystemExit(f"no CAMUS mount found under {INPUT}")


def nnunet_env(results: Path) -> dict[str, str]:
    env = {
        **os.environ,
        "nnUNet_raw": str(LOCAL / "nnUNet_raw"),
        "nnUNet_preprocessed": str(LOCAL / "nnUNet_preprocessed"),
        "nnUNet_results": str(results),
        # nnU-Net defaults to 12 augmentation workers; Colab has two cores and
        # the laptop smoke must not fork a dozen copies of the data
        "nnUNet_n_proc_DA": str(min(os.cpu_count() or 2, 4)),
    }
    return env


def complete(pred_dir: Path, n_cases: int) -> bool:
    # nnU-Net writes its json files before predicting, so only the mask count
    # says whether an interrupted prediction run finished
    return pred_dir.is_dir() and len(list(pred_dir.glob("*.nii.gz"))) == n_cases


def set_smoke_batch_size(preprocessed: Path) -> None:
    # the planned 2d batch size is sized for a GPU; the CPU smoke only checks
    # that every step runs, so two images per step are enough
    plans_path = preprocessed / "nnUNetPlans.json"
    plans = json.loads(plans_path.read_text())
    plans["configurations"][CONFIG]["batch_size"] = 2
    plans_path.write_text(json.dumps(plans, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", default="",
                        help="comma-separated, default all four (p05 only with --smoke)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()
    default = "p05" if args.smoke else ",".join(FOLD_OF_BUDGET)
    budgets = [b for b in (args.budgets or default).split(",") if b]
    unknown = set(budgets) - set(FOLD_OF_BUDGET)
    if unknown:
        raise SystemExit(f"unknown budgets {sorted(unknown)}")
    trainer = SMOKE_TRAINER if args.smoke else TRAINER

    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "-q", REPO_DIR, "nnunetv2"])
    data_root = find_data_root()
    results = WORK / "nnunet_results"
    env = nnunet_env(results)
    raw = LOCAL / "nnUNet_raw" / DATASET_NAME
    preprocessed = LOCAL / "nnUNet_preprocessed" / DATASET_NAME
    if not (raw / "splits_final.json").exists():
        cmd = [sys.executable, REPO_DIR / "scripts" / "prepare_nnunet.py", "--data-root",
               data_root, "--raw-dir", LOCAL / "nnUNet_raw",
               "--subsets-dir", REPO_DIR / "configs" / "subsets"]
        if args.smoke:
            cmd += ["--limit", "2"]
        run(cmd, cwd=REPO_DIR)
    if not (preprocessed / "nnUNetPlans.json").exists():
        run([tool("nnUNetv2_plan_and_preprocess"), "-d", DATASET_ID, "-c", CONFIG,
             "--verify_dataset_integrity"], env=env)
        if args.smoke:
            set_smoke_batch_size(preprocessed)
    # nnU-Net reads the split file from the preprocessed folder only
    shutil.copy(raw / "splits_final.json", preprocessed / "splits_final.json")

    model_dir = results / DATASET_NAME / f"{trainer}__nnUNetPlans__{CONFIG}"
    for budget in budgets:
        fold = FOLD_OF_BUDGET[budget]
        fold_dir = model_dir / f"fold_{fold}"
        if not (fold_dir / "checkpoint_final.pth").exists():
            cmd = [tool("nnUNetv2_train"), DATASET_ID, CONFIG, str(fold), "-tr", trainer,
                   "-device", args.device]
            if (fold_dir / "checkpoint_latest.pth").exists():
                cmd.append("--c")
            run(cmd, env=env)
        preds = WORK / "nnunet_preds" / budget
        n_test = len(list((raw / "imagesTs").glob("*.nii.gz")))
        for variant, extra in (("mirror", []), ("nomirror", ["--disable_tta"])):
            out = preds / variant
            if not complete(out, n_test):
                run([tool("nnUNetv2_predict"), "-i", raw / "imagesTs", "-o", out,
                     "-d", DATASET_ID, "-c", CONFIG, "-f", str(fold), "-tr", trainer,
                     "-device", args.device, *extra], env=env)
        cv_dir = model_dir / f"crossval_results_folds_{fold}"
        if not (cv_dir / "postprocessing.pkl").exists():
            run([tool("nnUNetv2_find_best_configuration"), DATASET_ID, "-c", CONFIG,
                 "-tr", trainer, "-f", str(fold), "--disable_ensembling"], env=env)
        for variant, out_name, flags in (
            ("mirror", f"nnunet_{budget}", []),
            ("nomirror", f"nnunet_nomirror_{budget}", ["--no-mirroring"]),
        ):
            pp = preds / f"{variant}_pp"
            if not complete(pp, n_test):
                run([tool("nnUNetv2_apply_postprocessing"), "-i", preds / variant, "-o", pp,
                     "-pp_pkl_file", cv_dir / "postprocessing.pkl",
                     "-plans_json", cv_dir / "plans.json",
                     "-dataset_json", cv_dir / "dataset.json"], env=env)
            cmd = [sys.executable, REPO_DIR / "scripts" / "export_nnunet.py", "--pred-dir", pp,
                   "--out-dir", WORK / "outputs" / out_name, "--data-root", data_root,
                   "--fold", str(fold), "--trainer", trainer, *flags]
            if args.smoke:
                cmd += ["--limit", "8"]
            run(cmd, cwd=REPO_DIR)


if __name__ == "__main__":
    main()
