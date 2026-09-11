"""Evaluate a trained checkpoint on the official test split.

    python -m medseg_label_efficiency.evaluate --config configs/camus_unet2d.yaml

Follows the CAMUS challenge protocol: the network predicts at its training
resolution, the prediction is resized back to the image's original grid, and
metrics are computed there against the untouched ground truth — Dice as
overlap, HD95 in millimeters using the pixel spacing from the NIfTI header.

Writes per-sample metrics (CSV), a summary with per-structure Dice/HD95 and
a breakdown by image quality (JSON), and prints the summary. The same metrics
module is used by the SAM 2.1 / MedSAM2 arms.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.data.utils import affine_to_spacing
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

from medseg_label_efficiency.config import load_config
from medseg_label_efficiency.data.registry import get_dataset_module
from medseg_label_efficiency.metrics import MetricAccumulator
from medseg_label_efficiency.reporting import quality_breakdown, write_metrics_files
from medseg_label_efficiency.train import build_model, pick_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/camus_unet2d.yaml")
    parser.add_argument("--checkpoint", default="", help="default: <out_dir>/best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0, help="truncate the split (smoke runs)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = get_dataset_module(cfg["dataset"])
    device = pick_device(args.device)
    out_dir = Path(cfg["out_dir"])
    ckpt_path = Path(args.checkpoint) if args.checkpoint else out_dir / "best.pt"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_model(cfg, max(ds.STRUCTURES) + 1).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    samples = ds.build_samples(cfg["data_root"], args.split)
    if args.limit:
        samples = samples[: args.limit]
    input_transform = ds.get_transforms(train=False, image_size=tuple(cfg["image_size"]))
    raw_transform = Compose(
        [LoadImaged(keys=["label"]), EnsureChannelFirstd(keys=["label"])]
    )

    acc = MetricAccumulator(ds.STRUCTURES, hausdorff=True)
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
                meta=[{k: sample[k] for k in ds.META_KEYS}],
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
    write_metrics_files(out_dir, args.split, summary, records)


if __name__ == "__main__":
    main()
