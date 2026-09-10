"""Zero-shot evaluation of promptable models (SAM 2.1, MedSAM2) on CAMUS.

    python -m medseg_label_efficiency.eval_sam --model sam2 --prompt box

For every test image and every structure, a prompt is simulated from the
ground truth (see prompts.py — oracle upper bound, single shot, no
refinement clicks), the model predicts one mask, and that mask is scored as
a binary problem on the native grid with HD95 in millimeters. Output format
matches evaluate.py so the arms are directly comparable.

The control is SAM2.1 hiera-tiny because MedSAM2 is a fine-tune of exactly
that variant: same architecture, same capacity, the only difference is
medical fine-tuning.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from monai.data.utils import affine_to_spacing
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

from medseg_label_efficiency.data.camus import build_samples
from medseg_label_efficiency.metrics import STRUCTURES, binary_scores
from medseg_label_efficiency.prompts import box_from_mask, point_from_mask, sample_rng

CHECKPOINTS = {
    "sam2": ("checkpoints/sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "medsam2": ("checkpoints/MedSAM2_latest.pt", "configs/sam2.1/sam2.1_hiera_t512.yaml"),
}


class Sam2Backend:
    """Thin wrapper so tests can swap in a mock. predict() takes an RGB uint8
    image plus one prompt and returns one boolean mask."""

    def __init__(self, model: str, device: str):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        ckpt, config = CHECKPOINTS[model]
        self._ensure_medsam2_config(config)
        sam_model = build_sam2(config, ckpt, device=torch.device(device))
        self.predictor = SAM2ImagePredictor(sam_model)
        # SAM2ImagePredictor hardcodes backbone feature sizes for 1024-pixel
        # input; MedSAM2 runs at 512, so the three pyramid levels are halved
        size = sam_model.image_size
        if size != 1024:
            self.predictor._bb_feat_sizes = [
                (size // 4, size // 4),
                (size // 8, size // 8),
                (size // 16, size // 16),
            ]

    @staticmethod
    def _ensure_medsam2_config(config: str) -> None:
        # MedSAM2 was fine-tuned at image size 512 with a config that is not
        # part of the sam2 package; download_checkpoints.sh fetches it and we
        # drop it into the installed package's config tree so hydra finds it.
        if not config.endswith("t512.yaml"):
            return
        import sam2 as sam2_pkg

        target = Path(sam2_pkg.__file__).parent / config
        source = Path("checkpoints/sam2.1_hiera_t512.yaml")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source.read_text())

    def predict(self, image_rgb, box=None, point=None):
        self.predictor.set_image(image_rgb)
        kwargs = {}
        if box is not None:  # (r0, c0, r1, c1) -> sam's (x0, y0, x1, y1)
            r0, c0, r1, c1 = box
            kwargs["box"] = np.array([c0, r0, c1, r1])
        if point is not None:  # (row, col) -> sam's (x, y)
            kwargs["point_coords"] = np.array([[point[1], point[0]]])
            kwargs["point_labels"] = np.array([1])
        masks, scores, _ = self.predictor.predict(**kwargs, multimask_output=True)
        return masks[int(np.argmax(scores))].astype(bool)


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    lo, hi = float(image.min()), float(image.max())
    scaled = (image - lo) / (hi - lo) if hi > lo else np.zeros_like(image)
    return np.stack([(scaled * 255).astype(np.uint8)] * 3, axis=-1)


def evaluate_model(backend, samples: list[dict], prompt_type: str) -> list[dict]:
    """One record per image: per-structure dice/hd95 plus metadata."""
    keys = ["image", "label"]
    load = Compose([LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys)])
    records = []
    for sample in samples:
        data = load({"image": sample["image"], "label": sample["label"]})
        image = data["image"][0].numpy()
        gt_map = data["label"][0].numpy().astype(np.int64)
        spacing = affine_to_spacing(data["label"].affine)[:2]
        spacing = (float(spacing[0]), float(spacing[1]))
        rgb = to_rgb_uint8(image)

        row = {k: sample[k] for k in ("patient", "view", "phase", "quality")}
        for structure_id, name in STRUCTURES.items():
            gt = gt_map == structure_id
            if prompt_type == "box":
                prompt = box_from_mask(gt, rng=sample_rng(sample["patient"], structure_id))
                pred = backend.predict(rgb, box=prompt) if prompt else np.zeros_like(gt)
            else:
                prompt = point_from_mask(gt)
                pred = backend.predict(rgb, point=prompt) if prompt else np.zeros_like(gt)
            dice, hd95 = binary_scores(pred, gt, spacing)
            row[f"dice_{name}"] = dice
            row[f"hd95_{name}"] = hd95
        records.append(row)
    return records


def summarize(records: list[dict]) -> dict:
    out = {}
    for key in records[0]:
        if key.startswith(("dice_", "hd95_")):
            out[key] = float(np.nanmean([r[key] for r in records]))
    out["dice_mean"] = float(np.nanmean([out[f"dice_{n}"] for n in STRUCTURES.values()]))
    out["hd95_mean"] = float(np.nanmean([out[f"hd95_{n}"] for n in STRUCTURES.values()]))
    groups = defaultdict(list)
    for r in records:
        groups[r.get("quality", "Unknown")].append(r)
    out["by_quality"] = {
        q: {
            "n": len(rows),
            **{
                f"dice_{n}": float(np.nanmean([r[f"dice_{n}"] for r in rows]))
                for n in STRUCTURES.values()
            },
        }
        for q, rows in sorted(groups.items())
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["sam2", "medsam2"], required=True)
    parser.add_argument("--prompt", choices=["box", "point"], required=True)
    parser.add_argument("--data-root", default="data/camus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", default="")
    parser.add_argument(
        "--decoder-weights", default="",
        help="load a fine-tuned mask-decoder checkpoint over the stock model",
    )
    args = parser.parse_args()

    samples = build_samples(args.data_root, args.split)
    if args.limit:
        samples = samples[: args.limit]

    backend = Sam2Backend(args.model, args.device)
    if args.decoder_weights:
        import torch

        ckpt = torch.load(args.decoder_weights, map_location=args.device, weights_only=True)
        backend.predictor.model.sam_mask_decoder.load_state_dict(ckpt["decoder"])
        print(f"loaded fine-tuned decoder from {args.decoder_weights} (step {ckpt['step']})")
    records = evaluate_model(backend, samples, args.prompt)

    out_dir = Path(args.out_dir or f"outputs/{args.model}_{args.prompt}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "prompt": args.prompt,
        "split": args.split,
        "n_samples": len(records),
        "protocol": "zero-shot, oracle prompts from GT (single shot), per-structure "
        "binary scoring on native grid, HD95 in mm",
        **summarize(records),
    }
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
