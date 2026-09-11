"""Train the supervised 2D U-Net baseline on CAMUS.

    python -m medseg_label_efficiency.train --config configs/camus_unet2d.yaml

Checkpoints go to <out_dir>/latest.pt (every epoch, for resuming after
preemption) and <out_dir>/best.pt (best validation Dice). Interrupted runs
continue with --resume, including their MLflow run.
"""

import argparse
import time
from pathlib import Path

import mlflow
import torch
from monai.data import DataLoader
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.utils import set_determinism

from medseg_label_efficiency.config import load_config
from medseg_label_efficiency.data.registry import get_dataset_module
from medseg_label_efficiency.metrics import MetricAccumulator


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: dict, num_classes: int) -> UNet:
    m = cfg["model"]
    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=num_classes,
        channels=tuple(m["channels"]),
        strides=tuple(m["strides"]),
        num_res_units=m["num_res_units"],
    )


def validate(model, loader, device, structures) -> float:
    model.eval()
    acc = MetricAccumulator(structures, hausdorff=False)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            pred = torch.argmax(model(images), dim=1, keepdim=True)
            acc.add(pred.cpu(), labels.cpu())
    return acc.summary()["dice_mean"]


def save_checkpoint(path, model, optimizer, scaler, epoch, best_dice, run_id):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_dice": best_dice,
            "mlflow_run_id": run_id,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/camus_unet2d.yaml")
    parser.add_argument("--resume", action="store_true", help="continue from <out_dir>/latest.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=0, help="override config epochs")
    parser.add_argument("--limit-train", type=int, default=0, help="truncate train set (smoke)")
    parser.add_argument("--limit-val", type=int, default=0, help="truncate val set (smoke runs)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = get_dataset_module(cfg["dataset"])
    tr = cfg["training"]
    epochs = args.epochs or tr["epochs"]
    device = pick_device(args.device)
    set_determinism(cfg["seed"])

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "latest.pt"
    best_path = out_dir / "best.pt"

    image_size = tuple(cfg["image_size"])
    cache = 0.0 if args.limit_train else cfg.get("cache_rate", 0.0)
    subset = None
    if cfg.get("train_subset"):
        subset = [
            line.strip()
            for line in Path(cfg["train_subset"]).read_text().splitlines()
            if line.strip()
        ]
        print(f"training restricted to {len(subset)} patients from {cfg['train_subset']}")
    train_ds = ds.get_dataset(
        cfg["data_root"], "train", image_size, cache, args.limit_train, patients=subset
    )
    val_ds = ds.get_dataset(cfg["data_root"], "val", image_size, cache, args.limit_val)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"]
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])

    model = build_model(cfg, max(ds.STRUCTURES) + 1).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    use_amp = bool(tr.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    start_epoch, best_dice, run_id = 0, 0.0, None
    if args.resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt["best_dice"]
        run_id = ckpt["mlflow_run_id"]
        print(f"resuming from epoch {start_epoch} (best val dice so far {best_dice:.4f})")

    mlflow.set_experiment("camus_unet2d")
    with mlflow.start_run(run_id=run_id) as run:
        if run_id is None:
            mlflow.log_params(
                {
                    "epochs": epochs,
                    "batch_size": cfg["batch_size"],
                    "image_size": image_size,
                    "lr": tr["lr"],
                    "weight_decay": tr["weight_decay"],
                    "channels": cfg["model"]["channels"],
                    "seed": cfg["seed"],
                    "train_samples": len(train_ds),
                    "device": device.type,
                    "amp": use_amp,
                }
            )

        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss, t0 = 0.0, time.time()
            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad()
                if use_amp:
                    with torch.autocast("cuda"):
                        loss = loss_fn(model(images), labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = loss_fn(model(images), labels)
                    loss.backward()
                    optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_loader)
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            msg = f"epoch {epoch + 1}/{epochs} loss {epoch_loss:.4f} ({time.time() - t0:.1f}s)"

            if (epoch + 1) % tr["val_interval"] == 0 or epoch == epochs - 1:
                val_dice = validate(model, val_loader, device, ds.STRUCTURES)
                mlflow.log_metric("val_dice", val_dice, step=epoch)
                msg += f" val dice {val_dice:.4f}"
                if val_dice > best_dice:
                    best_dice = val_dice
                    save_checkpoint(
                        best_path, model, optimizer, scaler, epoch, best_dice, run.info.run_id
                    )
                    msg += " (new best)"

            save_checkpoint(
                latest_path, model, optimizer, scaler, epoch, best_dice, run.info.run_id
            )
            print(msg)

        mlflow.log_metric("best_val_dice", best_dice)
    print(f"done, best val dice {best_dice:.4f}, checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
