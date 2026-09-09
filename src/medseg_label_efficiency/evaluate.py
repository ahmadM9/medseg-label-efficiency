"""Evaluate a trained checkpoint on the official CAMUS test split.

    python -m medseg_label_efficiency.evaluate --config configs/camus_unet2d.yaml

Writes per-sample metrics (CSV), a summary with per-structure Dice/HD95 and
a breakdown by image quality (JSON), and prints the summary. The same metrics
module is used later for the SAM 2.1 / MedSAM2 arms.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader

from medseg_label_efficiency.config import load_config
from medseg_label_efficiency.data.camus import get_dataset
from medseg_label_efficiency.metrics import MetricAccumulator
from medseg_label_efficiency.train import build_model, pick_device

META_KEYS = ("patient", "view", "phase", "quality")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/camus_unet2d.yaml")
    parser.add_argument("--checkpoint", default="", help="default: <out_dir>/best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0, help="truncate the split (smoke runs)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device(args.device)
    out_dir = Path(cfg["out_dir"])
    ckpt_path = Path(args.checkpoint) if args.checkpoint else out_dir / "best.pt"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = get_dataset(cfg["data_root"], args.split, tuple(cfg["image_size"]), 0.0, args.limit)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])

    acc = MetricAccumulator(hausdorff=True)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            pred = torch.argmax(model(images), dim=1, keepdim=True)
            meta = [
                {k: batch[k][i] for k in META_KEYS if k in batch}
                for i in range(images.shape[0])
            ]
            acc.add(pred.cpu(), labels, meta)

    records = acc.records()
    summary = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "n_samples": len(records),
        **acc.summary(),
        "by_quality": quality_breakdown(records),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.split}_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / f"{args.split}_per_sample.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}/{args.split}_metrics.json and _per_sample.csv")


if __name__ == "__main__":
    main()
