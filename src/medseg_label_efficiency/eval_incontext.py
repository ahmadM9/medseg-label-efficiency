"""Evaluate an in-context segmentation model on the official test split.

    python -m medseg_label_efficiency.eval_incontext --config configs/universeg_p05.yaml

The labelled budget subset is the support pool. For every test image and
every structure, K supports are drawn from the pool, the model predicts the
structure from those examples alone, and the mask is scored as a binary
problem on the native grid with HD95 in millimeters, the same path as the
prompted arms. No weight is updated and nothing is tuned on validation data.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.data.utils import affine_to_spacing
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

from medseg_label_efficiency.config import load_config
from medseg_label_efficiency.data.registry import get_dataset_module
from medseg_label_efficiency.eval_sam import summarize
from medseg_label_efficiency.incontext import (
    IN_CONTEXT_MODELS,
    build_backend,
    draw_supports,
    filter_pool,
    gray_uint8,
    load_pool,
    to_square,
)
from medseg_label_efficiency.metrics import binary_scores
from medseg_label_efficiency.prompts import sample_rng
from medseg_label_efficiency.reporting import write_metrics_files
from medseg_label_efficiency.train import pick_device


def evaluate_model(
    backend,
    queries: list[dict],
    pool: list[dict],
    pool_images: np.ndarray,
    pool_labels: np.ndarray,
    ds,
    shots: int,
    draws: int,
    match_keys: list[str],
) -> list[dict]:
    size = pool_images.shape[-1]
    keys = ["image", "label"]
    load = Compose([LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys)])
    records = []
    for query in queries:
        data = load({"image": query["image"], "label": query["label"]})
        gt_map = data["label"][0].numpy().astype(np.int64)
        spacing = affine_to_spacing(data["label"].affine)[:2]
        spacing = (float(spacing[0]), float(spacing[1]))
        image = to_square(gray_uint8(data["image"][0].numpy()), size, "bilinear")
        candidates = filter_pool(pool, query, match_keys)

        row = {k: query[k] for k in ds.META_KEYS}
        masks = []
        for structure_id, name in ds.STRUCTURES.items():
            # one seed per image and structure, so every structure gets its
            # own support draws and every run gets the same ones
            rng = sample_rng(*(query[k] for k in ds.META_KEYS), structure_id)
            preds = []
            for _ in range(draws):
                idx = draw_supports(candidates, shots, rng)
                preds.append(
                    backend.predict_draw(image, pool_images[idx], pool_labels[idx] == structure_id)
                )
            pred = torch.from_numpy(backend.combine(preds))[None, None].float()
            pred = F.interpolate(pred, size=gt_map.shape, mode="nearest")[0, 0].bool().numpy()
            dice, hd95 = binary_scores(pred, gt_map == structure_id, spacing)
            row[f"dice_{name}"] = dice
            row[f"hd95_{name}"] = hd95
            masks.append(pred)
        # per-structure binary tasks may claim the same pixel twice, which
        # the argmax arms cannot do; measured rather than assumed away
        row["overlap_frac"] = float((np.sum(masks, axis=0) >= 2).mean())
        records.append(row)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--shots", type=int, default=0,
        help="override the config's support size; output goes to <out_dir>_k<shots>",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = get_dataset_module(cfg["dataset"])
    device = pick_device(args.device)
    entry = IN_CONTEXT_MODELS[cfg["model"]]
    shots = args.shots or cfg["shots"]
    draws = cfg["draws"]
    match_keys = list(cfg.get("match_keys", []))
    out_dir = Path(cfg["out_dir"] + (f"_k{args.shots}" if args.shots else ""))

    pool = ds.build_samples(cfg["data_root"], "train")
    if cfg.get("train_subset"):
        keep = set(Path(cfg["train_subset"]).read_text().split())
        pool = [s for s in pool if s["patient"] in keep]
    queries = ds.build_samples(cfg["data_root"], args.split)
    if args.limit:
        queries = queries[: args.limit]
    n_candidates = min(len(filter_pool(pool, q, match_keys)) for q in queries)
    if shots > n_candidates:
        raise SystemExit(
            f"{shots} supports requested but some query has only {n_candidates} candidates "
            f"in the pool; lower --shots or use a larger budget"
        )

    print(f"loading {len(pool)} pool images at {entry['input_size']} px", flush=True)
    pool_images, pool_labels = load_pool(pool, entry["input_size"])
    backend = build_backend(cfg["model"], str(device))
    records = evaluate_model(
        backend, queries, pool, pool_images, pool_labels, ds, shots, draws, match_keys
    )

    rule = (
        "sigmoid mean over draws > 0.5"
        if entry["kind"] == "universeg"
        else "nearest palette colour per draw, majority over draws"
    )
    summary = {
        "model": cfg["model"],
        "split": args.split,
        "n_samples": len(records),
        "shots": shots,
        "draws": draws,
        "match_keys": match_keys,
        "n_pool": len(pool),
        "n_candidates_min": int(n_candidates),
        "protocol": "in-context, no gradient steps, no prompt at test time; "
        f"{shots} supports per draw from the budget subset only, matched on "
        f"{'+'.join(match_keys) or 'nothing'}, query patient excluded, drawn without "
        f"replacement with a fixed seed per image and structure; {draws} draws, {rule}; "
        "prediction resized to native grid; per-structure binary scoring, HD95 in mm; "
        "empty prediction vs non-empty GT scores Dice 0, both empty excluded, HD95 "
        "excluded when either mask is empty; structures may overlap (overlap_frac); "
        + entry["training_data"],
        **summarize(records, ds),
        "overlap_mean": float(np.mean([r["overlap_frac"] for r in records])),
    }
    write_metrics_files(out_dir, args.split, summary, records)


if __name__ == "__main__":
    main()
