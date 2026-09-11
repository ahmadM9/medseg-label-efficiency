"""Render prompt + prediction overlays for the promptable-model arms.

    python scripts/preview_prompts.py --model medsam2 --n 4

One PNG per (sample, structure): the image with the drawn prompt (box or
point), the predicted mask in red, and the ground-truth contour in green.
For eyeball verification that prompts land correctly and predictions are
sane before spending GPU time on the full test set.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

from medseg_label_efficiency.data.registry import get_dataset_module
from medseg_label_efficiency.eval_sam import to_rgb_uint8
from medseg_label_efficiency.promptable import Sam2Backend
from medseg_label_efficiency.prompts import box_from_mask, point_from_mask, sample_rng


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="medsam2", help="a promptable-model registry name")
    parser.add_argument("--dataset", default="camus")
    parser.add_argument("--data-root", default="data/camus")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="outputs/previews_prompts")
    parser.add_argument("--decoder-weights", default="", help="fine-tuned decoder checkpoint")
    args = parser.parse_args()

    ds = get_dataset_module(args.dataset)
    samples = ds.build_samples(args.data_root, "test")[: args.n]
    backend = Sam2Backend(args.model, args.device)
    if args.decoder_weights:
        import torch

        ckpt = torch.load(args.decoder_weights, map_location=args.device, weights_only=True)
        backend.predictor.model.sam_mask_decoder.load_state_dict(ckpt["decoder"])
    keys = ["image", "label"]
    load = Compose([LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys)])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        data = load({"image": sample["image"], "label": sample["label"]})
        image = data["image"][0].numpy()
        gt_map = data["label"][0].numpy().astype(np.int64)
        rgb = to_rgb_uint8(image)

        for structure_id, name in ds.STRUCTURES.items():
            gt = gt_map == structure_id
            fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
            styles = [("box", "yellow"), ("point", "cyan")]
            for ax, (prompt_type, color) in zip(axes, styles, strict=True):
                if prompt_type == "box":
                    prompt = box_from_mask(gt, rng=sample_rng(sample["patient"], structure_id))
                    pred = backend.predict(rgb, box=prompt)
                else:
                    prompt = point_from_mask(gt)
                    pred = backend.predict(rgb, point=prompt)
                ax.imshow(np.rot90(image), cmap="gray")
                ax.imshow(np.rot90(np.ma.masked_where(~pred, pred)), cmap="autumn", alpha=0.4)
                ax.contour(np.rot90(gt), colors="lime", linewidths=1)
                h = image.shape[1]  # rot90 maps (r, c) -> (h-1-c, r)
                if prompt_type == "box" and prompt is not None:
                    r0, c0, r1, c1 = prompt
                    ax.add_patch(
                        mpatches.Rectangle(
                            (r0, h - 1 - c1), r1 - r0, c1 - c0,
                            fill=False, edgecolor=color, linewidth=1.5,
                        )
                    )
                elif prompt is not None:
                    ax.plot(prompt[0], h - 1 - prompt[1], "o", color=color, markersize=6)
                ax.set_title(f"{prompt_type} prompt")
                ax.axis("off")
            fig.suptitle(f"{sample['patient']} {sample['view']} {sample['phase']} — {name} "
                         f"({args.model}, pred=red, GT=green)")
            fig.tight_layout()
            path = out_dir / f"{sample['patient']}_{sample['view']}_{sample['phase']}_{name}.png"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
