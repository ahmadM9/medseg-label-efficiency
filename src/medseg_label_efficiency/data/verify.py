"""Check an on-disk CAMUS copy for completeness and split consistency.

Run after downloading locally or rsyncing to a cluster:

    python -m medseg_label_efficiency.data.verify --data-root data/camus
"""

import argparse
import sys
from pathlib import Path

from medseg_label_efficiency.data.camus import PHASES, SPLIT_FILES, VIEWS, read_split

EXPECTED_PATIENTS = 500
EXPECTED_SPLIT_SIZES = {"train": 400, "val": 50, "test": 50}


def verify(data_root: str | Path) -> list[str]:
    """Return a list of problems; empty means the copy looks complete."""
    data_root = Path(data_root)
    problems = []

    nifti_dir = data_root / "database_nifti"
    split_dir = data_root / "database_split"
    if not nifti_dir.is_dir():
        return [f"missing directory: {nifti_dir}"]
    if not split_dir.is_dir():
        return [f"missing directory: {split_dir}"]

    patients = sorted(p.name for p in nifti_dir.iterdir() if p.name.startswith("patient"))
    if len(patients) != EXPECTED_PATIENTS:
        problems.append(f"expected {EXPECTED_PATIENTS} patients, found {len(patients)}")

    splits = {}
    for split in SPLIT_FILES:
        try:
            splits[split] = read_split(data_root, split)
        except FileNotFoundError:
            problems.append(f"missing split file: {SPLIT_FILES[split]}")
    if len(splits) == len(SPLIT_FILES):
        for split, ids in splits.items():
            if len(ids) != EXPECTED_SPLIT_SIZES[split]:
                problems.append(
                    f"{split} split: expected {EXPECTED_SPLIT_SIZES[split]}, found {len(ids)}"
                )
        all_ids = [pid for ids in splits.values() for pid in ids]
        if len(all_ids) != len(set(all_ids)):
            problems.append("splits overlap: some patients appear in more than one split")
        missing = set(all_ids) - set(patients)
        if missing:
            examples = sorted(missing)[:3]
            problems.append(f"{len(missing)} split patients missing on disk, e.g. {examples}")

    for patient in patients:
        pdir = nifti_dir / patient
        for view in VIEWS:
            if not (pdir / f"Info_{view}.cfg").is_file():
                problems.append(f"{patient}: missing Info_{view}.cfg")
            for phase in PHASES:
                for suffix in ("", "_gt"):
                    f = pdir / f"{patient}_{view}_{phase}{suffix}.nii.gz"
                    if not f.is_file():
                        problems.append(f"{patient}: missing {f.name}")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/camus")
    args = parser.parse_args()

    problems = verify(args.data_root)
    if problems:
        print(f"FAILED — {len(problems)} problem(s):")
        for p in problems[:20]:
            print(f"  - {p}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        sys.exit(1)
    print(f"OK — CAMUS copy at {args.data_root} is complete "
          f"({EXPECTED_PATIENTS} patients, splits 400/50/50 consistent)")


if __name__ == "__main__":
    main()
