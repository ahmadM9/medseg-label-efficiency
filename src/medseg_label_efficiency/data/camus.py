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

# label values in the ground-truth masks
LABELS = {0: "background", 1: "lv_endo", 2: "lv_myo", 3: "left_atrium"}

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


def get_dataset(
    data_root: str | Path,
    split: str,
    image_size: tuple[int, int] = (256, 256),
    cache_rate: float = 0.0,
    limit: int = 0,
) -> Dataset:
    """MONAI Dataset over the labeled ED/ES frames of one split.

    cache_rate > 0 switches to a CacheDataset (samples decoded once, kept in
    RAM). limit > 0 truncates the sample list, for smoke runs and debugging.
    """
    samples = build_samples(data_root, split)
    if limit > 0:
        samples = samples[:limit]
    transform = get_transforms(train=(split == "train"), image_size=image_size)
    if cache_rate > 0:
        return CacheDataset(data=samples, transform=transform, cache_rate=cache_rate)
    return Dataset(data=samples, transform=transform)
