"""Zero-shot / fine-tuned evaluation of promptable models on a dataset.

    python -m medseg_label_efficiency.eval_sam --model sam2 --prompt box

For every test image and every structure, a prompt is simulated from the
ground truth (see prompts.py: oracle upper bound, single shot, no
refinement clicks), the model predicts one mask, and that mask is scored as
a binary problem on the native grid with HD95 in millimeters. Output format
matches evaluate.py so the arms are directly comparable.

--box-source <dir> replaces the oracle boxes by tight boxes around the masks
another arm saved in <dir>/masks/ (the automatic-prompt cascade): no jitter,
and a structure the source did not predict gets no prompt and an empty mask.
--save-masks writes the combined label map per image to <out_dir>/masks/.

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
from medseg_label_efficiency.reporting import (
    combine_binary_masks,
    load_mask,
    quality_breakdown,
    save_mask,
    write_metrics_files,
)


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    lo, hi = float(image.min()), float(image.max())
    scaled = (image - lo) / (hi - lo) if hi > lo else np.zeros_like(image)
    return np.stack([(scaled * 255).astype(np.uint8)] * 3, axis=-1)


def evaluate_model(
    backend,
    samples: list[dict],
    prompt_type: str,
    ds,
    jitter_px: int = 5,
    box_source: str | Path | None = None,
    save_dir: str | Path | None = None,
) -> list[dict]:
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
        # the cascade prompts from another arm's prediction; scoring stays
        # against the ground truth either way
        prompt_map = load_mask(box_source, sample) if box_source else gt_map

        row = {k: sample[k] for k in ds.META_KEYS}
        masks = {}
        for structure_id, name in ds.STRUCTURES.items():
            gt = gt_map == structure_id
            source = prompt_map == structure_id
            if prompt_type == "box":
                # one seed per image and structure, so the four images of a
                # patient get their own jitter offsets
                rng = sample_rng(*(sample[k] for k in ds.META_KEYS), structure_id)
                prompt = box_from_mask(source, jitter_px=0 if box_source else jitter_px, rng=rng)
                pred = backend.predict(rgb, box=prompt) if prompt else np.zeros_like(gt)
            else:
                prompt = point_from_mask(source)
                pred = backend.predict(rgb, point=prompt) if prompt else np.zeros_like(gt)
            dice, hd95 = binary_scores(pred, gt, spacing)
            row[f"dice_{name}"] = dice
            row[f"hd95_{name}"] = hd95
            masks[structure_id] = pred
        # one prompt per structure means two masks may claim the same pixel,
        # which the argmax arms cannot do; measured rather than assumed away
        label_map, row["overlap_frac"] = combine_binary_masks(masks)
        if save_dir is not None:
            save_mask(save_dir, sample, label_map)
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
    if "overlap_frac" in records[0]:
        out["overlap_mean"] = float(np.mean([r["overlap_frac"] for r in records]))
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
    parser.add_argument(
        "--box-source", default="",
        help="out_dir of another arm whose saved masks provide the boxes (cascade)",
    )
    parser.add_argument("--save-masks", action="store_true", help="write masks to <out_dir>/masks/")
    args = parser.parse_args()
    if args.box_source and args.prompt != "box":
        parser.error("--box-source only applies to box prompts")

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

    out_dir = Path(args.out_dir or f"outputs/{args.model}_{args.prompt}")
    records = evaluate_model(
        backend, samples, args.prompt, ds, jitter_px=args.jitter_px,
        box_source=args.box_source or None, save_dir=out_dir if args.save_masks else None,
    )

    if args.box_source:
        jitter = 0
        protocol = (
            f"cascade: boxes from {args.box_source} (tight box around the saved prediction, "
            "no jitter; a structure the source did not predict gets an empty mask), "
            "per-structure binary scoring on native grid, HD95 in mm"
        )
    else:
        jitter = args.jitter_px if args.prompt == "box" else None
        protocol = (
            "zero-shot, oracle prompts from GT (single shot), per-structure "
            "binary scoring on native grid, HD95 in mm"
        )
    summary = {
        "model": args.model,
        "prompt": args.prompt,
        "split": args.split,
        "n_samples": len(records),
        "jitter_px": jitter,
        "box_source": args.box_source or None,
        "decoder_weights": args.decoder_weights or None,
        "protocol": protocol,
        **summarize(records, ds),
    }
    write_metrics_files(out_dir, args.split, summary, records)


if __name__ == "__main__":
    main()
