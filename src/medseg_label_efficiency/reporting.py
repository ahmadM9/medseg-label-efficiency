"""Shared result reporting: the quality breakdown and the JSON/CSV writers
every evaluation CLI uses, so all arms produce identical file formats."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def quality_breakdown(records: list[dict]) -> dict:
    """Mean Dice per structure within each image-quality group."""
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
