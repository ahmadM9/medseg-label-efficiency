"""Segmentation metrics shared by every arm of the comparison.

All evaluations (supervised U-Net, SAM 2.1, MedSAM2) go through the same
accumulator so the numbers are comparable by construction: per-structure
Dice and 95th-percentile Hausdorff distance, computed per sample.
"""

import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.utils import one_hot

NUM_CLASSES = 4
STRUCTURES = {1: "lv_endo", 2: "lv_myo", 3: "left_atrium"}


class MetricAccumulator:
    """Collects per-sample metrics from batches of integer label maps.

    Predictions and ground truth arrive as (B, 1, H, W) tensors of class
    indices. HD95 is optional because it is slow and not needed for
    validation-time model selection.
    """

    def __init__(self, hausdorff: bool = True):
        self.dice = DiceMetric(include_background=False, reduction="none")
        self.hd95 = (
            HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")
            if hausdorff
            else None
        )
        self.meta: list[dict] = []

    def add(self, pred: torch.Tensor, gt: torch.Tensor, meta: list[dict] | None = None) -> None:
        pred_oh = one_hot(pred, NUM_CLASSES)
        gt_oh = one_hot(gt, NUM_CLASSES)
        self.dice(pred_oh, gt_oh)
        if self.hd95 is not None:
            self.hd95(pred_oh, gt_oh)
        if meta is not None:
            self.meta.extend(meta)

    def records(self) -> list[dict]:
        """One dict per sample: metadata plus dice/hd95 for each structure."""
        dice = self.dice.get_buffer().cpu().numpy()
        hd = self.hd95.get_buffer().cpu().numpy() if self.hd95 is not None else None
        rows = []
        for i in range(dice.shape[0]):
            row = dict(self.meta[i]) if i < len(self.meta) else {}
            for j, name in enumerate(STRUCTURES.values()):
                row[f"dice_{name}"] = float(dice[i, j])
                if hd is not None:
                    row[f"hd95_{name}"] = float(hd[i, j])
            rows.append(row)
        return rows

    def summary(self) -> dict[str, float]:
        """Mean per structure and overall. An infinite HD95 (empty prediction
        or empty ground truth for a structure) is excluded from the mean."""
        out = {}
        dice = self.dice.get_buffer().cpu().numpy()
        for j, name in enumerate(STRUCTURES.values()):
            out[f"dice_{name}"] = float(np.nanmean(dice[:, j]))
        out["dice_mean"] = float(np.nanmean(dice))
        if self.hd95 is not None:
            hd = self.hd95.get_buffer().cpu().numpy()
            hd = np.where(np.isinf(hd), np.nan, hd)
            for j, name in enumerate(STRUCTURES.values()):
                out[f"hd95_{name}"] = float(np.nanmean(hd[:, j]))
            out["hd95_mean"] = float(np.nanmean(hd))
        return out

    def reset(self) -> None:
        self.dice.reset()
        if self.hd95 is not None:
            self.hd95.reset()
        self.meta = []
