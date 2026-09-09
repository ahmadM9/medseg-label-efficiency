"""Correctness tests for the shared metrics module."""

import torch

from medseg_label_efficiency.metrics import MetricAccumulator


def _label_map(size=64):
    gt = torch.zeros(1, 1, size, size, dtype=torch.long)
    gt[..., 10:30, 10:30] = 1
    gt[..., 35:50, 10:30] = 2
    gt[..., 10:30, 35:55] = 3
    return gt


def test_perfect_prediction():
    gt = _label_map()
    acc = MetricAccumulator(hausdorff=True)
    acc.add(gt.clone(), gt)
    s = acc.summary()
    assert s["dice_lv_endo"] == 1.0
    assert s["dice_mean"] == 1.0
    assert s["hd95_mean"] == 0.0


def test_known_overlap_dice():
    # square shifted by half its width: overlap 200 of 400+400 -> dice 0.5
    gt = torch.zeros(1, 1, 64, 64, dtype=torch.long)
    gt[..., 10:30, 10:30] = 1
    pred = torch.zeros_like(gt)
    pred[..., 10:30, 20:40] = 1
    acc = MetricAccumulator(hausdorff=False)
    acc.add(pred, gt)
    assert abs(acc.summary()["dice_lv_endo"] - 0.5) < 1e-6


def test_missing_structure_scores_zero_dice():
    gt = _label_map()
    pred = gt.clone()
    pred[pred == 3] = 0
    acc = MetricAccumulator(hausdorff=False)
    acc.add(pred, gt)
    s = acc.summary()
    assert s["dice_left_atrium"] == 0.0
    assert s["dice_lv_endo"] == 1.0


def test_records_carry_metadata():
    gt = _label_map()
    acc = MetricAccumulator(hausdorff=False)
    acc.add(gt.clone(), gt, meta=[{"patient": "patient0001", "quality": "Good"}])
    rows = acc.records()
    assert len(rows) == 1
    assert rows[0]["patient"] == "patient0001"
    assert rows[0]["dice_lv_myo"] == 1.0
