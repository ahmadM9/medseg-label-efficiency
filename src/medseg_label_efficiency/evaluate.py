"""Evaluate a trained checkpoint on the official CAMUS test split.

    python -m medseg_label_efficiency.evaluate --config configs/camus_unet2d.yaml

Follows the CAMUS challenge protocol: the network predicts at its training
resolution, the prediction is resized back to the image's original grid, and
metrics are computed there against the untouched ground truth — Dice as
overlap, HD95 in millimeters using the pixel spacing from the NIfTI header.

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
import torch.nn.functional as F
from monai.data.utils import affine_to_spacing
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

from medseg_label_efficiency.config import load_config
from medseg_label_efficiency.data.camus import build_samples, get_transforms
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

    samples = build_samples(cfg["data_root"], args.split)
    if args.limit:
        samples = samples[: args.limit]
    input_transform = get_transforms(train=False, image_size=tuple(cfg["image_size"]))
    raw_transform = Compose(
        [LoadImaged(keys=["label"]), EnsureChannelFirstd(keys=["label"])]
    )

    acc = MetricAccumulator(hausdorff=True)
    with torch.no_grad():
        for sample in samples:
            image = input_transform(dict(sample))["image"].unsqueeze(0).to(device)
            gt = raw_transform({"label": sample["label"]})["label"]
            spacing = affine_to_spacing(gt.affine)[:2]
            pred = torch.argmax(model(image), dim=1, keepdim=True).float().cpu()
            pred = F.interpolate(pred, size=gt.shape[-2:], mode="nearest")
            acc.add(
                pred.long(),
                gt.as_tensor().long().unsqueeze(0),
                meta=[{k: sample[k] for k in META_KEYS}],
                spacing=(float(spacing[0]), float(spacing[1])),
            )

    records = acc.records()
    summary = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "n_samples": len(records),
        "protocol": "prediction resized to native grid; HD95 in mm; "
        "Dice mean excludes background",
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
