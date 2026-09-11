"""Train a segmentation head on a frozen encoder at a label budget.

    python -m medseg_label_efficiency.train_head --config configs/dino2_head_p05.yaml

The encoder (encoders.py registry) never trains; each image's patch-token
grid is computed once and cached, so head steps are cheap — the same trick
finetune_sam.py uses. Unlike the promptable arms this one is FULLY
AUTOMATIC: multi-class output, no prompts, same loss family as the U-Net.
Loss and validation live in the encoder's square input space; the official
native-grid evaluation is eval_head.py.
"""

import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged
from monai.utils import set_determinism

from medseg_label_efficiency.config import load_config
from medseg_label_efficiency.data.registry import get_dataset_module
from medseg_label_efficiency.encoders import FrozenEncoder
from medseg_label_efficiency.eval_sam import to_rgb_uint8
from medseg_label_efficiency.metrics import MetricAccumulator
from medseg_label_efficiency.seg_head import build_head
from medseg_label_efficiency.train import pick_device


def precompute(encoder: FrozenEncoder, samples: list[dict]) -> list[dict]:
    """Frozen features + square-resized GT label map, cached per image."""
    keys = ["image", "label"]
    load = Compose([LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys)])
    size = encoder.input_size
    entries = []
    for sample in samples:
        data = load({"image": sample["image"], "label": sample["label"]})
        rgb = to_rgb_uint8(data["image"][0].numpy())
        gt_map = torch.from_numpy(data["label"][0].numpy().astype(np.int64))
        # uint8 keeps the cache small (label values are 0..num_classes-1);
        # batches cast to long only when they hit the loss/metric
        gt_sq = F.interpolate(
            gt_map[None, None].float(), size=(size, size), mode="nearest"
        )[0, 0].to(torch.uint8)
        entries.append(
            {
                "feat": encoder.embed(rgb).to(torch.float16),
                "gt": gt_sq,
                "patient": sample["patient"],
            }
        )
    return entries


@torch.no_grad()
def validate(head, entries, structures, device, batch_size, out_size) -> float:
    acc = MetricAccumulator(structures, hausdorff=False)
    for start in range(0, len(entries), batch_size):
        chunk = entries[start : start + batch_size]
        feats = torch.stack([e["feat"] for e in chunk]).to(device, torch.float32)
        preds = torch.argmax(head(feats, out_size), dim=1, keepdim=True).cpu()
        gts = torch.stack([e["gt"] for e in chunk]).unsqueeze(1).long()
        acc.add(preds, gts)
    return acc.summary()["dice_mean"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=0, help="override config steps")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = get_dataset_module(cfg["dataset"])
    tr = cfg["training"]
    steps_total = args.steps or tr["steps"]
    set_determinism(cfg["seed"])
    device = pick_device(args.device)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path, best_path = out_dir / "head_latest.pt", out_dir / "head_best.pt"

    encoder = FrozenEncoder(cfg["encoder"], str(device))
    num_classes = max(ds.STRUCTURES) + 1
    head = build_head(encoder.dim, num_classes).to(device)

    train_samples = ds.build_samples(cfg["data_root"], "train")
    if cfg.get("train_subset"):
        keep = set(Path(cfg["train_subset"]).read_text().split())
        train_samples = [s for s in train_samples if s["patient"] in keep]
        print(f"training restricted to {len(keep)} patients")
    val_samples = ds.build_samples(cfg["data_root"], "val")

    print(f"caching embeddings: {len(train_samples)} train + {len(val_samples)} val images")
    train_entries = precompute(encoder, train_samples)
    val_entries = precompute(encoder, val_samples)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"]
    )
    rng = np.random.default_rng(cfg["seed"])
    batch_size = cfg["batch_size"]
    out_size = encoder.input_size

    start_step, best_dice, run_id = 0, 0.0, None
    if args.resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=True)
        head.load_state_dict(ckpt["head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_dice, run_id = ckpt["step"], ckpt["best_dice"], ckpt["mlflow_run_id"]
        print(f"resuming from step {start_step} (best val dice {best_dice:.4f})")

    def save(path, step, run):
        torch.save(
            {
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best_dice": best_dice,
                "encoder": cfg["encoder"],
                "mlflow_run_id": run.info.run_id,
            },
            path,
        )

    mlflow.set_experiment("frozen_encoder_head")
    with mlflow.start_run(run_id=run_id) as run:
        if run_id is None:
            mlflow.log_params(
                {
                    "encoder": cfg["encoder"], "steps": steps_total, "lr": tr["lr"],
                    "batch_size": batch_size, "seed": cfg["seed"],
                    "train_patients": len({s["patient"] for s in train_samples}),
                    "device": device.type,
                }
            )
        head.train()
        t0, running = time.time(), 0.0
        for step in range(start_step, steps_total):
            idx = rng.choice(len(train_entries), size=batch_size)
            feats = torch.stack(
                [train_entries[i]["feat"] for i in idx]
            ).to(device, torch.float32)
            gts = (
                torch.stack([train_entries[i]["gt"] for i in idx])
                .unsqueeze(1).long().to(device)
            )
            logits = head(feats, out_size)
            loss = loss_fn(logits, gts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

            if (step + 1) % tr["val_interval"] == 0 or step + 1 == steps_total:
                head.eval()
                val_dice = validate(
                    head, val_entries, ds.STRUCTURES, device, batch_size, out_size
                )
                head.train()
                running_avg = running / tr["val_interval"]
                running = 0.0
                mlflow.log_metric("train_loss", running_avg, step=step + 1)
                mlflow.log_metric("val_dice", val_dice, step=step + 1)
                marker = ""
                if val_dice > best_dice:
                    best_dice = val_dice
                    save(best_path, step + 1, run)
                    marker = " (new best)"
                save(latest_path, step + 1, run)
                print(
                    f"step {step + 1}/{steps_total} loss {running_avg:.4f} "
                    f"val dice {val_dice:.4f}{marker} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
        mlflow.log_metric("best_val_dice", best_dice)
    print(f"done, best val dice {best_dice:.4f}, checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
