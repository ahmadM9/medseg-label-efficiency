"""Zero-shot / fine-tuned evaluation of promptable models on a dataset.

    python -m medseg_label_efficiency.eval_sam --model sam2 --prompt box

For every test image and every structure, a prompt is simulated from the
ground truth (see prompts.py — oracle upper bound, single shot, no
refinement clicks), the model predicts one mask, and that mask is scored as
a binary problem on the native grid with HD95 in millimeters. Output format
matches evaluate.py so the arms are directly comparable.

Models come from the promptable-model registry (promptable.py); datasets
from the dataset registry. The tiny SAM2.1 control exists because MedSAM2
is a fine-tune of exactly that variant: same architecture, same capacity,
the only difference is medical fine-tuning.
"""

import argparse
from pathlib import Path

import numpy as np
from monai.data.utils import affine_to_spacing
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

from medseg_label_efficiency.data.registry import get_dataset_module
from medseg_label_efficiency.metrics import binary_scores
from medseg_label_efficiency.promptable import Sam2Backend
from medseg_label_efficiency.prompts import box_from_mask, point_from_mask, sample_rng
from medseg_label_efficiency.reporting import quality_breakdown, write_metrics_files


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    lo, hi = float(image.min()), float(image.max())
    scaled = (image - lo) / (hi - lo) if hi > lo else np.zeros_like(image)
    return np.stack([(scaled * 255).astype(np.uint8)] * 3, axis=-1)


def evaluate_model(
    backend, samples: list[dict], prompt_type: str, ds, jitter_px: int = 5
) -> list[dict]:
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

        row = {k: sample[k] for k in ds.META_KEYS}
        for structure_id, name in ds.STRUCTURES.items():
            gt = gt_map == structure_id
            if prompt_type == "box":
                rng = sample_rng(sample["patient"], structure_id)
                prompt = box_from_mask(gt, jitter_px=jitter_px, rng=rng)
                pred = backend.predict(rgb, box=prompt) if prompt else np.zeros_like(gt)
            else:
                prompt = point_from_mask(gt)
                pred = backend.predict(rgb, point=prompt) if prompt else np.zeros_like(gt)
            dice, hd95 = binary_scores(pred, gt, spacing)
            row[f"dice_{name}"] = dice
            row[f"hd95_{name}"] = hd95
        records.append(row)
    return records


def summarize(records: list[dict], ds) -> dict:
    out = {}
    for key in records[0]:
        if key.startswith(("dice_", "hd95_")):
            out[key] = float(np.nanmean([r[key] for r in records]))
    names = list(ds.STRUCTURES.values())
    out["dice_mean"] = float(np.nanmean([out[f"dice_{n}"] for n in names]))
    out["hd95_mean"] = float(np.nanmean([out[f"hd95_{n}"] for n in names]))
    out["by_quality"] = quality_breakdown(records)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="a promptable-model registry name")
    parser.add_argument("--prompt", choices=["box", "point"], required=True)
    parser.add_argument("--dataset", default="camus")
    parser.add_argument("--data-root", default="data/camus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", default="")
    parser.add_argument(
        "--jitter-px", type=int, default=5,
        help="box edge jitter in pixels; 0 gives the exact tight box (prompt ablation)",
    )
    parser.add_argument(
        "--decoder-weights", default="",
        help="load a fine-tuned mask-decoder checkpoint over the stock model",
    )
    args = parser.parse_args()

    ds = get_dataset_module(args.dataset)
    samples = ds.build_samples(args.data_root, args.split)
    if args.limit:
        samples = samples[: args.limit]

    backend = Sam2Backend(args.model, args.device)
    if args.decoder_weights:
        import torch

        ckpt = torch.load(args.decoder_weights, map_location=args.device, weights_only=True)
        backend.predictor.model.sam_mask_decoder.load_state_dict(ckpt["decoder"])
        print(f"loaded fine-tuned decoder from {args.decoder_weights} (step {ckpt['step']})")
    records = evaluate_model(backend, samples, args.prompt, ds, jitter_px=args.jitter_px)

    out_dir = Path(args.out_dir or f"outputs/{args.model}_{args.prompt}")
    summary = {
        "model": args.model,
        "prompt": args.prompt,
        "split": args.split,
        "n_samples": len(records),
        "jitter_px": args.jitter_px if args.prompt == "box" else None,
        "protocol": "zero-shot, oracle prompts from GT (single shot), per-structure "
        "binary scoring on native grid, HD95 in mm",
        **summarize(records, ds),
    }
    write_metrics_files(out_dir, args.split, summary, records)


if __name__ == "__main__":
    main()
