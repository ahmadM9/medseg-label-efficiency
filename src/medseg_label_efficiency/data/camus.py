"""CAMUS dataset access.

Expects the NIfTI release of CAMUS as shipped by the Human Heart Project
platform, reachable under ``data_root``:

    database_nifti/patientXXXX/patientXXXX_{2CH,4CH}_{ED,ES}[_gt].nii.gz
    database_nifti/patientXXXX/Info_{2CH,4CH}.cfg
    database_split/subgroup_{training,validation,testing}.txt

The official train/val/test split (400/50/50 patients, Leclerc et al.,
TMI 2019) is read from the split files verbatim — the test set is a
scattered list of patient IDs, not a contiguous range.
"""

from pathlib import Path

from monai.data import CacheDataset, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandRotated,
    RandZoomd,
    Resized,
    ScaleIntensityd,
)

SPLIT_FILES = {
    "train": "subgroup_training.txt",
    "val": "subgroup_validation.txt",
    "test": "subgroup_testing.txt",
}
SPLITS = tuple(SPLIT_FILES)

# label values in the ground-truth masks
LABELS = {0: "background", 1: "lv_endo", 2: "lv_myo", 3: "left_atrium"}

# the foreground structures, as the rest of the pipeline consumes them
# (anatomy belongs to the dataset module, never to metrics/reporting code)
STRUCTURES = {1: "lv_endo", 2: "lv_myo", 3: "left_atrium"}
STRUCTURE_TITLES = {
    "lv_endo": "LV endocardium",
    "lv_myo": "LV myocardium",
    "left_atrium": "Left atrium",
}

# structures scored as the union of several labels. lv_epi is the
# CAMUS-challenge "LV epicardium": the filled region inside the epicardial
# contour, cavity plus wall, which is what published CAMUS LVEpi numbers
# (the MedSAM2 paper included) refer to. Not the same task as lv_myo.
DERIVED_STRUCTURES = {"lv_epi": (1, 2)}

# per-sample metadata keys carried into evaluation records/CSVs
META_KEYS = ("patient", "view", "phase", "quality")

VIEWS = ("2CH", "4CH")
PHASES = ("ED", "ES")

# some hosting platforms (Kaggle) transparently gunzip uploads, so a copy may
# hold plain .nii files instead of the .nii.gz the platform ships
NII_EXTENSIONS = (".nii.gz", ".nii")


def find_nii(directory: Path, stem: str) -> Path:
    for ext in NII_EXTENSIONS:
        path = directory / f"{stem}{ext}"
        if path.is_file():
            return path
    return directory / f"{stem}{NII_EXTENSIONS[0]}"


def read_split(data_root: str | Path, split: str) -> list[str]:
    """Return the patient IDs of one official split ("train"/"val"/"test")."""
    path = Path(data_root) / "database_split" / SPLIT_FILES[split]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def parse_info(path: str | Path) -> dict:
    """Parse a per-view Info_{2CH,4CH}.cfg file ("Key: value" lines)."""
    info: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    return info


def build_samples(data_root: str | Path, split: str) -> list[dict]:
    """List one sample dict per labeled image (patient x view x cardiac phase).

    Each dict carries the image/label paths plus metadata used later for
    quality-stratified evaluation.
    """
    data_root = Path(data_root)
    samples = []
    for patient in read_split(data_root, split):
        pdir = data_root / "database_nifti" / patient
        for view in VIEWS:
            info = parse_info(pdir / f"Info_{view}.cfg")
            for phase in PHASES:
                samples.append(
                    {
                        "image": str(find_nii(pdir, f"{patient}_{view}_{phase}")),
                        "label": str(find_nii(pdir, f"{patient}_{view}_{phase}_gt")),
                        "patient": patient,
                        "view": view,
                        "phase": phase,
                        "quality": info.get("ImageQuality", "Unknown"),
                        "ef": float(info["EF"]) if "EF" in info else None,
                    }
                )
    return samples


def build_sequences(data_root: str | Path, split: str) -> list[dict]:
    """One dict per (patient, view) half-cycle sequence, ED to ES, labeled on
    every frame. Frame 0 is ED and the last frame is ES."""
    data_root = Path(data_root)
    sequences = []
    for patient in read_split(data_root, split):
        pdir = data_root / "database_nifti" / patient
        for view in VIEWS:
            info = parse_info(pdir / f"Info_{view}.cfg")
            sequences.append(
                {
                    "image": str(find_nii(pdir, f"{patient}_{view}_half_sequence")),
                    "label": str(find_nii(pdir, f"{patient}_{view}_half_sequence_gt")),
                    "patient": patient,
                    "view": view,
                    "quality": info.get("ImageQuality", "Unknown"),
                    "n_frames": int(info["NbFrame"]) if "NbFrame" in info else None,
                }
            )
    return sequences


def get_transforms(train: bool, image_size: tuple[int, int] = (256, 256)) -> Compose:
    """Deterministic loading chain, plus augmentations for training.

    Images keep their native intensity range until min-max scaling (no 8-bit
    quantization anywhere). No horizontal flips: they would mirror the heart's
    chirality, which is anatomically meaningless for apical views.
    """
    transforms: list = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys="image"),
        Resized(keys="image", spatial_size=image_size, mode="bilinear"),
        Resized(keys="label", spatial_size=image_size, mode="nearest"),
    ]
    if train:
        transforms += [
            RandRotated(
                keys=["image", "label"],
                range_x=0.17,  # ~10 degrees
                prob=0.5,
                mode=["bilinear", "nearest"],
            ),
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.9,
                max_zoom=1.1,
                prob=0.3,
                mode=["bilinear", "nearest"],
            ),
            RandGaussianNoised(keys="image", prob=0.2, std=0.02),
            RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.8, 1.25)),
        ]
    transforms.append(EnsureTyped(keys=["image", "label"]))
    return Compose(transforms)


EXPECTED_PATIENTS = 500
EXPECTED_SPLIT_SIZES = {"train": 400, "val": 50, "test": 50}


def verify(data_root: str | Path) -> list[str]:
    """Check an on-disk CAMUS copy; returns problems, empty means complete."""
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
                    stem = f"{patient}_{view}_{phase}{suffix}"
                    if not find_nii(pdir, stem).is_file():
                        problems.append(f"{patient}: missing {stem}.nii[.gz]")

    return problems


def get_dataset(
    data_root: str | Path,
    split: str,
    image_size: tuple[int, int] = (256, 256),
    cache_rate: float = 0.0,
    limit: int = 0,
    patients: list[str] | None = None,
) -> Dataset:
    """MONAI Dataset over the labeled ED/ES frames of one split.

    cache_rate > 0 switches to a CacheDataset (samples decoded once, kept in
    RAM). limit > 0 truncates the sample list, for smoke runs and debugging.
    patients restricts to a patient list (the label-efficiency subsets).
    """
    samples = build_samples(data_root, split)
    if patients is not None:
        keep = set(patients)
        samples = [s for s in samples if s["patient"] in keep]
    if limit > 0:
        samples = samples[:limit]
    transform = get_transforms(train=(split == "train"), image_size=image_size)
    if cache_rate > 0:
        return CacheDataset(data=samples, transform=transform, cache_rate=cache_rate)
    return Dataset(data=samples, transform=transform)
