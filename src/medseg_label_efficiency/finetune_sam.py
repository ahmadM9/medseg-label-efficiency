"""Fine-tune the MedSAM2 mask decoder on a label-budget subset.

    python -m medseg_label_efficiency.finetune_sam --config configs/medsam2_ft_p05.yaml

Recipe: image encoder and prompt encoder stay frozen; only the mask decoder
(~4M params) trains. Training samples are (cached image embedding, jittered
ground-truth box, structure mask); the jitter is redrawn every step, acting
as augmentation. Because the encoder is frozen, every image's features are
computed once and cached, so decoder steps are cheap.

Geometry note: sam2 resizes inputs to a 512x512 square (not aspect-
preserving), so ground-truth masks are nearest-resized to that same square
and the loss lives in that space. Test-set evaluation goes through the
normal eval_sam.py native-grid protocol instead.
"""

import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import DiceLoss
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged
from monai.utils import set_determinism

from medseg_label_efficiency.config import load_config
from medseg_label_efficiency.data.camus import build_samples
from medseg_label_efficiency.eval_sam import Sam2Backend, to_rgb_uint8
from medseg_label_efficiency.metrics import STRUCTURES
from medseg_label_efficiency.prompts import box_from_mask, sample_rng

MODEL_SIZE = 512  # MedSAM2's input resolution


def freeze_all_but_decoder(model: torch.nn.Module) -> int:
    """Only sam_mask_decoder trains; returns the trainable-parameter count."""
    trainable = 0
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("sam_mask_decoder")
        trainable += param.numel() if param.requires_grad else 0
    return trainable


def build_pairs(entries: list[dict]) -> list[tuple[int, int]]:
    """(image index, structure id) pairs, skipping structures absent from GT."""
    pairs = []
    for i, entry in enumerate(entries):
        for structure_id in STRUCTURES:
            if entry["gt512"][structure_id - 1].any():
                pairs.append((i, structure_id))
    return pairs


def precompute(backend: Sam2Backend, samples: list[dict], device) -> list[dict]:
    """Run the frozen encoder once per image; cache features + 512-square GT."""
    keys = ["image", "label"]
    load = Compose([LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys)])
    entries = []
    for sample in samples:
        data = load({"image": sample["image"], "label": sample["label"]})
        rgb = to_rgb_uint8(data["image"][0].numpy())
        gt_map = torch.from_numpy(data["label"][0].numpy().astype(np.int64))
        gt512 = torch.stack(
            [
                F.interpolate(
                    (gt_map == s).float()[None, None], size=(MODEL_SIZE, MODEL_SIZE),
                    mode="nearest",
                )[0, 0].bool()
                for s in STRUCTURES
            ]
        )
        with torch.no_grad():
            backend.predictor.set_image(rgb)
        feats = backend.predictor._features
        entries.append(
            {
                "embed": feats["image_embed"][0].detach().to("cpu", torch.float16),
                "hr": [f[0].detach().to("cpu", torch.float16) for f in feats["high_res_feats"]],
                "gt512": gt512,
                "patient": sample["patient"],
            }
        )
    return entries


def decode_batch(model, entries, pairs, boxes, device):
    """Prompt-encode the boxes and run the mask decoder; returns 512-sized logits."""
    embed = torch.stack([entries[i]["embed"] for i, _ in pairs]).to(device, torch.float32)
    hr = [
        torch.stack([entries[i]["hr"][level] for i, _ in pairs]).to(device, torch.float32)
        for level in range(2)
    ]
    # (r0, c0, r1, c1) -> sam's corner points (x, y) with labels 2 (TL) and 3 (BR)
    coords = torch.tensor(
        [[[c0, r0], [c1, r1]] for (r0, c0, r1, c1) in boxes], dtype=torch.float32, device=device
    )
    labels = torch.tensor([[2, 3]], device=device).repeat(len(boxes), 1)
    sparse, dense = model.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
    low_res, _, _, _ = model.sam_mask_decoder(
        image_embeddings=embed,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        high_res_features=hr,
    )
    return F.interpolate(low_res, size=(MODEL_SIZE, MODEL_SIZE), mode="bilinear")


def draw_boxes(entries, pairs, jitter_px, rng=None, fixed_seed=False):
    boxes = []
    for i, structure_id in pairs:
        mask = entries[i]["gt512"][structure_id - 1].numpy()
        box_rng = (
            sample_rng(entries[i]["patient"], structure_id) if fixed_seed else rng
        )
        boxes.append(box_from_mask(mask, jitter_px=jitter_px, rng=box_rng))
    return boxes


@torch.no_grad()
def validate(model, entries, pairs, jitter_px, device, batch_size) -> float:
    dices = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        boxes = draw_boxes(entries, chunk, jitter_px, fixed_seed=True)
        logits = decode_batch(model, entries, chunk, boxes, device)
        preds = logits[:, 0] > 0
        for (i, structure_id), pred in zip(chunk, preds, strict=True):
            gt = entries[i]["gt512"][structure_id - 1].to(device)
            denom = pred.sum() + gt.sum()
            dices.append((2 * (pred & gt).sum() / denom).item() if denom else float("nan"))
    return float(np.nanmean(dices))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=0, help="override config steps")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tr = cfg["training"]
    steps_total = args.steps or tr["steps"]
    set_determinism(cfg["seed"])
    if args.device != "auto":
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path, best_path = out_dir / "decoder_latest.pt", out_dir / "decoder_best.pt"

    backend = Sam2Backend("medsam2", str(device))
    model = backend.predictor.model
    n_trainable = freeze_all_but_decoder(model)
    print(f"trainable decoder parameters: {n_trainable / 1e6:.2f}M")

    train_samples = build_samples(cfg["data_root"], "train")
    if cfg.get("train_subset"):
        keep = set(Path(cfg["train_subset"]).read_text().split())
        train_samples = [s for s in train_samples if s["patient"] in keep]
        print(f"training restricted to {len(keep)} patients")
    val_samples = build_samples(cfg["data_root"], "val")

    print(f"caching embeddings: {len(train_samples)} train + {len(val_samples)} val images")
    train_entries = precompute(backend, train_samples, device)
    val_entries = precompute(backend, val_samples, device)
    train_pairs = build_pairs(train_entries)
    val_pairs = build_pairs(val_entries)

    dice_loss = DiceLoss(sigmoid=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tr["lr"], weight_decay=tr["weight_decay"],
    )
    rng = np.random.default_rng(cfg["seed"])
    batch_size = cfg["batch_size"]
    jitter = cfg.get("jitter_px", 5)

    start_step, best_dice, run_id = 0, 0.0, None
    if args.resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=True)
        model.sam_mask_decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_dice, run_id = ckpt["step"], ckpt["best_dice"], ckpt["mlflow_run_id"]
        print(f"resuming from step {start_step} (best val dice {best_dice:.4f})")

    def save(path, step):
        torch.save(
            {
                "decoder": model.sam_mask_decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best_dice": best_dice,
                "mlflow_run_id": run.info.run_id,
            },
            path,
        )

    mlflow.set_experiment("medsam2_finetune")
    with mlflow.start_run(run_id=run_id) as run:
        if run_id is None:
            mlflow.log_params(
                {
                    "steps": steps_total, "lr": tr["lr"], "batch_size": batch_size,
                    "train_patients": len({s["patient"] for s in train_samples}),
                    "jitter_px": jitter, "seed": cfg["seed"], "device": device.type,
                }
            )
        model.sam_mask_decoder.train()
        t0, running = time.time(), 0.0
        for step in range(start_step, steps_total):
            idx = rng.choice(len(train_pairs), size=batch_size)
            chunk = [train_pairs[i] for i in idx]
            boxes = draw_boxes(train_entries, chunk, jitter, rng=rng)
            logits = decode_batch(model, train_entries, chunk, boxes, device)
            gts = torch.stack(
                [train_entries[i]["gt512"][s - 1] for i, s in chunk]
            ).to(device, torch.float32)[:, None]
            loss = dice_loss(logits, gts) + F.binary_cross_entropy_with_logits(logits, gts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

            if (step + 1) % tr["val_interval"] == 0 or step + 1 == steps_total:
                model.sam_mask_decoder.eval()
                val_dice = validate(model, val_entries, val_pairs, jitter, device, batch_size)
                model.sam_mask_decoder.train()
                running_avg = running / tr["val_interval"]
                running = 0.0
                mlflow.log_metric("train_loss", running_avg, step=step + 1)
                mlflow.log_metric("val_dice", val_dice, step=step + 1)
                marker = ""
                if val_dice > best_dice:
                    best_dice = val_dice
                    save(best_path, step + 1)
                    marker = " (new best)"
                save(latest_path, step + 1)
                print(
                    f"step {step + 1}/{steps_total} loss {running_avg:.4f} "
                    f"val dice {val_dice:.4f}{marker} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
        mlflow.log_metric("best_val_dice", best_dice)
    print(f"done, best val dice {best_dice:.4f}, checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
