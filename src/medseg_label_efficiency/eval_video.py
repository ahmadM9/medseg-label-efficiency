"""Video-mode versus still-mode evaluation of promptable models on sequences.

    python -m medseg_label_efficiency.eval_video --model medsam2 --mode video

Every frame of every half-cycle sequence carries a label. In video mode one
tight box per structure is drawn on the middle frame and the model propagates
the mask to the other frames through its memory; in still mode every frame is
prompted and predicted on its own with the image predictor. Same frames, same
boxes (no jitter), so the difference between the two modes is the propagation
effect alone. The dataset's derived structures (for CAMUS, the filled
lv_epi = cavity plus wall) are scored too, which gives the row that published
CAMUS "LV epicardium" numbers can be compared against.
"""

import argparse
from pathlib import Path

import numpy as np
from monai.data.utils import affine_to_spacing
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

from medseg_label_efficiency.data.registry import get_dataset_module
from medseg_label_efficiency.eval_sam import to_rgb_uint8
from medseg_label_efficiency.metrics import binary_scores
from medseg_label_efficiency.prompts import box_from_mask
from medseg_label_efficiency.reporting import quality_breakdown, write_metrics_files

META_KEYS = ("patient", "view", "quality")


def structure_names(ds) -> list[str]:
    return list(ds.STRUCTURES.values()) + list(getattr(ds, "DERIVED_STRUCTURES", {}))


def structure_masks(gt_frame: np.ndarray, ds) -> dict[str, np.ndarray]:
    masks = {name: gt_frame == label for label, name in ds.STRUCTURES.items()}
    for name, labels in getattr(ds, "DERIVED_STRUCTURES", {}).items():
        masks[name] = np.isin(gt_frame, labels)
    return masks


def load_sequence(sequence: dict):
    keys = ["image", "label"]
    load = Compose([LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys)])
    data = load({"image": sequence["image"], "label": sequence["label"]})
    frames = data["image"][0].numpy()  # (H, W, T)
    gt = data["label"][0].numpy().astype(np.int64)
    spacing = affine_to_spacing(data["label"].affine)[:2]
    return frames, gt, (float(spacing[0]), float(spacing[1]))


def evaluate_video(backend, sequences: list[dict], ds, mode: str, jitter_px: int = 0):
    """One record per frame: per-structure dice/hd95 plus sequence metadata."""
    names = structure_names(ds)
    records = []
    for sequence in sequences:
        frames, gt, spacing = load_sequence(sequence)
        n_frames = frames.shape[-1]
        rgb = [to_rgb_uint8(frames[..., t]) for t in range(n_frames)]
        gts = [structure_masks(gt[..., t], ds) for t in range(n_frames)]

        if mode == "video":
            # the paper's protocol: prompt the middle frame, propagate both ways
            prompt_frame = n_frames // 2
            boxes = {}
            for obj_id, name in enumerate(names):
                box = box_from_mask(gts[prompt_frame][name], jitter_px=jitter_px)
                if box is not None:
                    boxes[obj_id] = box
            tracked = backend.track(rgb, boxes, prompt_frame)
            preds = [
                {name: tracked.get(t, {}).get(obj_id) for obj_id, name in enumerate(names)}
                for t in range(n_frames)
            ]
        else:
            preds = []
            for t in range(n_frames):
                frame_pred = {}
                for name in names:
                    box = box_from_mask(gts[t][name], jitter_px=jitter_px)
                    frame_pred[name] = backend.predict(rgb[t], box=box) if box else None
                preds.append(frame_pred)

        for t in range(n_frames):
            row = {k: sequence[k] for k in META_KEYS}
            row.update(frame=t, n_frames=n_frames, is_ed=t == 0, is_es=t == n_frames - 1)
            for name in names:
                pred = preds[t][name]
                if pred is None:
                    pred = np.zeros_like(gts[t][name])
                dice, hd95 = binary_scores(pred, gts[t][name], spacing)
                row[f"dice_{name}"] = dice
                row[f"hd95_{name}"] = hd95
            records.append(row)
    return records


def _stats(records: list[dict], names: list[str]) -> dict:
    out = {}
    for name in names:
        dice = np.array([r[f"dice_{name}"] for r in records], dtype=float)
        hd = np.array([r[f"hd95_{name}"] for r in records], dtype=float)
        out[f"dice_{name}"] = float(np.nanmean(dice))
        out[f"dice_{name}_median"] = float(np.nanmedian(dice))
        out[f"dice_{name}_q25"] = float(np.nanpercentile(dice, 25))
        out[f"dice_{name}_q75"] = float(np.nanpercentile(dice, 75))
        out[f"hd95_{name}"] = float(np.nanmean(hd))
    return out


def summarize(records: list[dict], ds) -> dict:
    """Mean and median per structure over all frames and over ED/ES frames only.
    dice_mean covers the dataset's base structures so it lines up with the
    still-image rows; derived structures are reported on their own."""
    base = list(ds.STRUCTURES.values())
    names = structure_names(ds)
    out = _stats(records, names)
    out["dice_mean"] = float(np.nanmean([out[f"dice_{n}"] for n in base]))
    out["hd95_mean"] = float(np.nanmean([out[f"hd95_{n}"] for n in base]))
    ed_es = [r for r in records if r["is_ed"] or r["is_es"]]
    out["ed_es"] = _stats(ed_es, names)
    out["ed_es"]["dice_mean"] = float(np.nanmean([out["ed_es"][f"dice_{n}"] for n in base]))
    out["ed_es"]["n_frames"] = len(ed_es)
    out["by_quality"] = quality_breakdown(records)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="a promptable-model registry name")
    parser.add_argument("--mode", choices=["video", "still"], required=True)
    parser.add_argument("--dataset", default="camus")
    parser.add_argument("--data-root", default="data/camus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="number of sequences")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    ds = get_dataset_module(args.dataset)
    sequences = ds.build_sequences(args.data_root, args.split)
    if args.limit:
        sequences = sequences[: args.limit]

    if args.mode == "video":
        from medseg_label_efficiency.promptable import Sam2VideoBackend

        backend = Sam2VideoBackend(args.model, args.device)
        protocol = (
            "video mode: tight box (no jitter) per structure on the middle frame, "
            "propagated forward and backward by the model's memory; frames passed as "
            "JPEG q95; per-frame binary scoring on the native grid, HD95 in mm"
        )
    else:
        from medseg_label_efficiency.promptable import Sam2Backend

        backend = Sam2Backend(args.model, args.device)
        protocol = (
            "still mode: every frame prompted alone with a tight box (no jitter) per "
            "structure, image predictor; per-frame binary scoring on the native grid, "
            "HD95 in mm"
        )
    records = evaluate_video(backend, sequences, ds, args.mode)

    out_dir = Path(args.out_dir or f"outputs/{args.model}_{args.mode}_video")
    summary = {
        "model": args.model,
        "mode": args.mode,
        "split": args.split,
        "n_sequences": len(sequences),
        "n_frames": len(records),
        "protocol": protocol,
        **summarize(records, ds),
    }
    write_metrics_files(out_dir, args.split, summary, records)


if __name__ == "__main__":
    main()
