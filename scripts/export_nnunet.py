"""Score nnU-Net's predicted test NIfTIs through the shared metrics and mask store.

    python scripts/export_nnunet.py --pred-dir <nnUNet predictions> --out-dir outputs/nnunet_p05 \
        [--data-root data/camus] [--fold 0] [--trainer nnUNetTrainer_100epochs] \
        [--no-mirroring] [--no-postprocessing] [--limit N]

nnU-Net writes one label map per test case on the native grid. This reads
each one next to the untouched ground truth with the same loader as
evaluate.py, accumulates Dice and HD95 in millimetres per structure, writes
test_metrics.json, test_per_sample.csv and masks/ in the layout every other
arm uses, and records the nnU-Net settings in the protocol string. nnU-Net's
own metrics are never reported. A prediction with a shape different from its
ground truth is an export error and stops the run.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from monai.data.utils import affine_to_spacing
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

from medseg_label_efficiency.data.camus import META_KEYS, STRUCTURES, build_samples
from medseg_label_efficiency.metrics import MetricAccumulator
from medseg_label_efficiency.reporting import quality_breakdown, save_mask, write_metrics_files
from prepare_nnunet import case_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-root", default="data/camus")
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--trainer", default="nnUNetTrainer_100epochs")
    parser.add_argument("--no-mirroring", action="store_true")
    parser.add_argument("--no-postprocessing", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="first N test samples (smoke)")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    samples = build_samples(args.data_root, "test")
    if args.limit:
        samples = samples[: args.limit]
    keys = ["label", "pred"]
    load = Compose([LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys)])

    acc = MetricAccumulator(STRUCTURES, hausdorff=True)
    for sample in samples:
        pred_path = pred_dir / f"{case_id(sample)}.nii.gz"
        if not pred_path.exists():
            raise SystemExit(f"no prediction for {case_id(sample)} under {pred_dir}")
        loaded = load({"label": sample["label"], "pred": str(pred_path)})
        gt, pred = loaded["label"], loaded["pred"]
        if tuple(pred.shape) != tuple(gt.shape):
            raise SystemExit(f"{pred_path.name}: shape {tuple(pred.shape)} vs GT {tuple(gt.shape)}")
        pred = torch.as_tensor(np.rint(np.asarray(pred)), dtype=torch.long).unsqueeze(0)
        spacing = affine_to_spacing(gt.affine)[:2]
        acc.add(
            pred,
            gt.as_tensor().long().unsqueeze(0),
            meta=[{k: sample[k] for k in META_KEYS}],
            spacing=(float(spacing[0]), float(spacing[1])),
        )
        save_mask(out_dir, sample, pred[0, 0].numpy())

    records = acc.records()
    summary = {
        "predictions": str(pred_dir),
        "trainer": args.trainer,
        "fold": args.fold,
        "mirroring": not args.no_mirroring,
        "postprocessing": not args.no_postprocessing,
        "split": "test",
        "n_samples": len(records),
        "protocol": "nnU-Net v2 2d prediction on the native grid, "
        f"{'with' if not args.no_mirroring else 'without'} mirroring, "
        f"{'with' if not args.no_postprocessing else 'without'} post-processing; "
        "HD95 in mm; Dice mean excludes background",
        **acc.summary(),
        "by_quality": quality_breakdown(records),
    }
    write_metrics_files(out_dir, "test", summary, records)


if __name__ == "__main__":
    main()
