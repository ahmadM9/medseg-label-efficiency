import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# every evaluation CLI writes its results through these, so all arms produce
# the same file formats and the analysis scripts read one layout


def quality_breakdown(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in records:
        groups[row.get("quality", "Unknown")].append(row)
    out = {}
    for quality, rows in sorted(groups.items()):
        dice_cols = [k for k in rows[0] if k.startswith("dice_")]
        out[quality] = {
            "n": len(rows),
            **{col: float(np.nanmean([r[col] for r in rows])) for col in dice_cols},
        }
    return out


def write_metrics_files(out_dir: str | Path, split: str, summary: dict, records: list[dict]):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{split}_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / f"{split}_per_sample.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}/{split}_metrics.json and _per_sample.csv")


def mask_path(out_dir: str | Path, sample: dict) -> Path:
    return Path(out_dir) / "masks" / f"{sample['patient']}_{sample['view']}_{sample['phase']}.npz"


def save_mask(out_dir: str | Path, sample: dict, pred: np.ndarray) -> Path:
    # the predicted label map on the native image grid, uint8 and compressed:
    # about 5 KB per image, so a full test split of masks stays under 2 MB
    path = mask_path(out_dir, sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mask=np.asarray(pred, dtype=np.uint8))
    return path


def load_mask(out_dir: str | Path, sample: dict) -> np.ndarray:
    with np.load(mask_path(out_dir, sample)) as data:
        return data["mask"]


def combine_binary_masks(masks: dict[int, np.ndarray]) -> tuple[np.ndarray, float]:
    # per-structure binary predictions folded into one label map; where two
    # structures claim a pixel the lowest id wins, and the fraction of such
    # pixels is returned so the overlap is measured, not hidden
    shape = next(iter(masks.values())).shape
    label_map = np.zeros(shape, dtype=np.uint8)
    for structure_id in sorted(masks, reverse=True):
        label_map[masks[structure_id]] = structure_id
    overlap = float((np.sum(list(masks.values()), axis=0) >= 2).mean())
    return label_map, overlap
