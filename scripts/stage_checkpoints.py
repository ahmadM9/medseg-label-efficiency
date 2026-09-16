"""Copy the seed-0 checkpoints into a staging folder with the optimizer states stripped.

    python scripts/stage_checkpoints.py [--outputs outputs] [--staging outputs/staging]

The Kaggle downloads keep optimizer (and AMP scaler) states that a
re-evaluation never reads; dropping them cuts the upload from about 850 MB
to about 230 MB. The staging layout is outputs/<out_dir name>/<file>, the
same shape the kaggle/scale launcher globs for, so the folder is uploaded to
Drive as it is. Sources are the recorded download directories, never the
bare outputs/<arm>/ dirs, which have held local smoke leftovers before.
"""

import argparse
from pathlib import Path

import torch

# out_dir name -> (checkpoint file, download directory that holds the real run)
SOURCES = {
    "camus_unet2d_p05": ("best.pt", "kaggle-subsets"),
    "camus_unet2d_p10": ("best.pt", "kaggle-subsets"),
    "camus_unet2d_p25": ("best.pt", "kaggle-subsets"),
    "camus_unet2d": ("best.pt", "kaggle-run1"),
    "medsam2_ft_p05": ("decoder_best.pt", "kaggle-ft"),
    "medsam2_ft_p10": ("decoder_best.pt", "kaggle-ft"),
    "medsam2_ft_p25": ("decoder_best.pt", "kaggle-ft"),
    "medsam2_ft_full": ("decoder_best.pt", "kaggle-ft"),
    # the dino2 p05 run landed in the smoke kernel's download; same metrics as recorded
    "dino2_head_p05": ("head_best.pt", "kaggle-head-smoke"),
    "dino2_head_p10": ("head_best.pt", "kaggle-head-dino2"),
    "dino2_head_p25": ("head_best.pt", "kaggle-head-dino2"),
    "dino2_head_full": ("head_best.pt", "kaggle-head-dino2"),
    "dino3_head_p05": ("head_best.pt", "kaggle-head-dino3"),
    "dino3_head_p10": ("head_best.pt", "kaggle-head-dino3"),
    "dino3_head_p25": ("head_best.pt", "kaggle-head-dino3"),
    "dino3_head_full": ("head_best.pt", "kaggle-head-dino3"),
}
DROP = ("optimizer", "scaler")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--staging", default="outputs/staging")
    args = parser.parse_args()
    total = 0
    for name, (filename, download) in SOURCES.items():
        src = Path(args.outputs) / download / "outputs" / name / filename
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        slim = {k: v for k, v in ckpt.items() if k not in DROP}
        dst = Path(args.staging) / "outputs" / name / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(slim, dst)
        size = dst.stat().st_size
        total += size
        print(f"{name}/{filename}: {size / 1e6:.0f} MB, kept {sorted(slim)}")
    print(f"total {total / 1e6:.0f} MB under {args.staging}/outputs")


if __name__ == "__main__":
    main()
