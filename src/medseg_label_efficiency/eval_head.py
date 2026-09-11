"""Evaluate a frozen-encoder + trained-head model on the official test split.

    python -m medseg_label_efficiency.eval_head --config configs/dino2_head_p05.yaml

Fully automatic (no prompts): the model outputs all structures at once, so
this arm competes with the U-Net on identical terms. Same native-grid
protocol as evaluate.py: argmax prediction resized back to the original
image, Dice + HD95 in millimeters from the header spacing.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.data.utils import affine_to_spacing
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

from medseg_label_efficiency.config import load_config
from medseg_label_efficiency.data.registry import get_dataset_module
from medseg_label_efficiency.encoders import FrozenEncoder
from medseg_label_efficiency.eval_sam import to_rgb_uint8
from medseg_label_efficiency.metrics import MetricAccumulator
from medseg_label_efficiency.reporting import quality_breakdown, write_metrics_files
from medseg_label_efficiency.seg_head import build_head
from medseg_label_efficiency.train import pick_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="", help="default: <out_dir>/head_best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = get_dataset_module(cfg["dataset"])
    device = pick_device(args.device)
    out_dir = Path(cfg["out_dir"])
    ckpt_path = Path(args.checkpoint) if args.checkpoint else out_dir / "head_best.pt"

    encoder = FrozenEncoder(cfg["encoder"], str(device))
    head = build_head(encoder.dim, max(ds.STRUCTURES) + 1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    head.load_state_dict(ckpt["head"])
    head.eval()

    samples = ds.build_samples(cfg["data_root"], args.split)
    if args.limit:
        samples = samples[: args.limit]
    keys = ["image", "label"]
    load = Compose([LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys)])

    acc = MetricAccumulator(ds.STRUCTURES, hausdorff=True)
    with torch.no_grad():
        for sample in samples:
            data = load({"image": sample["image"], "label": sample["label"]})
            rgb = to_rgb_uint8(data["image"][0].numpy())
            gt = data["label"]
            spacing = affine_to_spacing(gt.affine)[:2]
            feats = encoder.embed(rgb)[None].to(device, torch.float32)
            pred = torch.argmax(head(feats, encoder.input_size), dim=1, keepdim=True)
            pred = F.interpolate(pred.float().cpu(), size=gt.shape[-2:], mode="nearest")
            acc.add(
                pred.long(),
                gt.as_tensor().long().unsqueeze(0),
                meta=[{k: sample[k] for k in ds.META_KEYS}],
                spacing=(float(spacing[0]), float(spacing[1])),
            )

    records = acc.records()
    summary = {
        "checkpoint": str(ckpt_path),
        "encoder": cfg["encoder"],
        "split": args.split,
        "n_samples": len(records),
        "protocol": "frozen encoder + trained head, fully automatic (no prompts); "
        "prediction resized to native grid; HD95 in mm; Dice mean excludes background",
        **acc.summary(),
        "by_quality": quality_breakdown(records),
    }
    write_metrics_files(out_dir, args.split, summary, records)


if __name__ == "__main__":
    main()
