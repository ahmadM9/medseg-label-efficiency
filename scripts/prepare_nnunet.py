"""Lay CAMUS out as an nnU-Net v2 raw dataset with one fold per label budget.

    python scripts/prepare_nnunet.py --data-root data/camus --raw-dir nnUNet_raw [--limit 2]

Writes nnUNet_raw/Dataset501_CAMUS/{imagesTr,labelsTr,imagesTs}, dataset.json
and splits_final.json. imagesTr holds the official train and val patients
(the val split is the validation set of every fold), imagesTs the official
test patients. Each label budget is a fold: fold 0 trains on the seed-0
20-patient subset, fold 1 on 40, fold 2 on 100, fold 3 on all 400, so one
preprocessing run serves four trainings and their results land in fold_0 to
fold_3. splits_final.json must be copied next to the preprocessed data
before training; nnU-Net reads it from there and not from the raw folder.

Images and labels are re-saved through nibabel: labels become uint8 (CAMUS
ships them as float32, nnU-Net wants integers) and both get a uniform .nii.gz
ending whatever the source copy uses. Affines are kept, so the predictions
come back on the native grid. --limit N keeps the first N patients of the
smallest subset, of val and of test, for a CPU smoke run.
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from medseg_label_efficiency.data.camus import PHASES, STRUCTURES, VIEWS, build_samples, read_split

DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID}_CAMUS"
# fold index -> subset list; None means the whole official train split.
# the lists are nested, so a --limit taken from the smallest one is in every fold
FOLDS = {0: "train_p05.txt", 1: "train_p10.txt", 2: "train_p25.txt", 3: None}
BUDGET_OF_FOLD = {0: "p05", 1: "p10", 2: "p25", 3: "full"}


def case_id(sample: dict) -> str:
    return f"CAMUS_{sample['patient']}_{sample['view']}_{sample['phase']}"


def parse_case_id(case: str) -> dict:
    _, patient, view, phase = case.split("_")
    if view not in VIEWS or phase not in PHASES:
        raise ValueError(f"not a CAMUS case id: {case}")
    return {"patient": patient, "view": view, "phase": phase}


def save_as(src: str, dst: Path, dtype=None) -> None:
    img = nib.load(src)
    data = np.asanyarray(img.dataobj)
    if dtype is not None:
        rounded = np.rint(data)
        if not np.array_equal(rounded, data):
            raise ValueError(f"non-integer label values in {src}")
        data = rounded.astype(dtype)
    out = nib.Nifti1Image(data, img.affine)
    out.set_data_dtype(data.dtype)
    nib.save(out, dst)


def write_cases(samples: list[dict], images_dir: Path, labels_dir: Path | None) -> list[str]:
    images_dir.mkdir(parents=True, exist_ok=True)
    if labels_dir is not None:
        labels_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for sample in samples:
        case = case_id(sample)
        save_as(sample["image"], images_dir / f"{case}_0000.nii.gz")
        if labels_dir is not None:
            save_as(sample["label"], labels_dir / f"{case}.nii.gz", dtype=np.uint8)
        cases.append(case)
    return cases


def select_patients(data_root: Path, subsets_dir: Path, limit: int) -> dict[str, list[str]]:
    train = read_split(data_root, "train")
    val = read_split(data_root, "val")
    test = read_split(data_root, "test")
    if limit:
        smallest = (subsets_dir / FOLDS[0]).read_text().split()[:limit]
        train = [p for p in train if p in smallest]
        val, test = val[:limit], test[:limit]
    return {"train": train, "val": val, "test": test}


def fold_splits(train_cases: list[str], val_cases: list[str], subsets_dir: Path) -> list[dict]:
    splits = []
    for fold, subset in FOLDS.items():
        if subset is None:
            keep = train_cases
        else:
            patients = set((subsets_dir / subset).read_text().split())
            keep = [c for c in train_cases if parse_case_id(c)["patient"] in patients]
        if not keep:
            raise SystemExit(f"fold {fold} ({BUDGET_OF_FOLD[fold]}) has no training cases")
        splits.append({"train": keep, "val": list(val_cases)})
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/camus")
    parser.add_argument("--raw-dir", default="nnUNet_raw")
    parser.add_argument("--subsets-dir", default="configs/subsets")
    parser.add_argument("--limit", type=int, default=0, help="patients per split (smoke)")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    subsets_dir = Path(args.subsets_dir)
    root = Path(args.raw_dir) / DATASET_NAME

    patients = select_patients(data_root, subsets_dir, args.limit)
    by_patient = {}
    for split in ("train", "val", "test"):
        for s in build_samples(data_root, split):
            by_patient.setdefault(s["patient"], []).append(s)
    samples = {k: [s for p in v for s in by_patient[p]] for k, v in patients.items()}

    train_cases = write_cases(samples["train"], root / "imagesTr", root / "labelsTr")
    val_cases = write_cases(samples["val"], root / "imagesTr", root / "labelsTr")
    write_cases(samples["test"], root / "imagesTs", None)

    labels = {"background": 0, **{name: idx for idx, name in STRUCTURES.items()}}
    dataset = {
        "name": "CAMUS",
        "description": "CAMUS 2D echocardiography, ED and ES frames of the 2CH and 4CH views",
        # any channel name other than CT gets nnU-Net's z-score normalisation
        "channel_names": {"0": "US"},
        "labels": labels,
        "numTraining": len(train_cases) + len(val_cases),
        "file_ending": ".nii.gz",
    }
    (root / "dataset.json").write_text(json.dumps(dataset, indent=2))
    splits = fold_splits(train_cases, val_cases, subsets_dir)
    (root / "splits_final.json").write_text(json.dumps(splits, indent=2))
    for fold, split in enumerate(splits):
        print(f"fold {fold} ({BUDGET_OF_FOLD[fold]}): {len(split['train'])} train, "
              f"{len(split['val'])} val")
    print(f"wrote {root}: {dataset['numTraining']} training images, "
          f"{len(samples['test'])} test images")


if __name__ == "__main__":
    main()
