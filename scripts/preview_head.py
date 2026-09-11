"""Render frozen-encoder head predictions vs ground truth for eyeballing.

    python scripts/preview_head.py --config configs/dino2_head_p05.yaml --n 4

One PNG per test image: predicted structures as filled color overlay,
ground-truth contours on top (same color language as preview_overlays.py).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

from medseg_label_efficiency.config import load_config
from medseg_label_efficiency.data.registry import get_dataset_module
from medseg_label_efficiency.encoders import FrozenEncoder
from medseg_label_efficiency.eval_sam import to_rgb_uint8
from medseg_label_efficiency.seg_head import build_head

OVERLAY = ListedColormap(["none", "tab:red", "tab:blue", "tab:green"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="", help="default: <out_dir>/head_best.pt")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="outputs/previews_head")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = get_dataset_module(cfg["dataset"])
    encoder = FrozenEncoder(cfg["encoder"], args.device)
    head = build_head(encoder.dim, max(ds.STRUCTURES) + 1).to(args.device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(cfg["out_dir"]) / "head_best.pt"
    head.load_state_dict(
        torch.load(ckpt_path, map_location=args.device, weights_only=True)["head"]
    )
    head.eval()

    samples = ds.build_samples(cfg["data_root"], "test")[: args.n]
    keys = ["image", "label"]
    load = Compose([LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys)])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        data = load({"image": sample["image"], "label": sample["label"]})
        image = data["image"][0].numpy()
        gt_map = data["label"][0].numpy().astype(np.int64)
        with torch.no_grad():
            feats = encoder.embed(to_rgb_uint8(image))[None].to(args.device, torch.float32)
            pred = torch.argmax(head(feats, encoder.input_size), dim=1, keepdim=True)
        pred = F.interpolate(pred.float(), size=image.shape, mode="nearest")[0, 0].numpy()

        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.imshow(np.rot90(image), cmap="gray")
        ax.imshow(
            np.rot90(np.ma.masked_equal(pred, 0)), cmap=OVERLAY, alpha=0.4, vmin=0, vmax=3
        )
        for structure_id in ds.STRUCTURES:
            ax.contour(np.rot90(gt_map == structure_id), colors="lime", linewidths=1)
        ax.axis("off")
        ax.set_title(
            f"{sample['patient']} {sample['view']} {sample['phase']} — "
            f"{cfg['encoder']} pred (fill) vs GT (green)"
        )
        fig.tight_layout()
        path = out_dir / f"{sample['patient']}_{sample['view']}_{sample['phase']}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
