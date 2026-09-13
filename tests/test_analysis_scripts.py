"""The paired-comparison and seed-aggregation scripts on synthetic results."""

import csv
import json

import numpy as np
import pytest

from aggregate_seeds import ARMS, aggregate
from compare_arms import compare, holm, per_patient, run_family

STRUCTURES = ("lv_endo", "lv_myo", "left_atrium")


def write_per_sample(path, dice_by_patient: dict[str, float], shift: float = 0.0, n_patients=50):
    # four images per patient, per-structure dice around the patient's level
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    fields = ["patient", "view", "phase", "quality"]
    fields += [f"{m}_{s}" for s in STRUCTURES for m in ("dice", "hd95")]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for patient, level in dice_by_patient.items():
            for view in ("2CH", "4CH"):
                for phase in ("ED", "ES"):
                    row = {"patient": patient, "view": view, "phase": phase, "quality": "Good"}
                    for s in STRUCTURES:
                        row[f"dice_{s}"] = min(1.0, level + shift + rng.normal(0, 0.01))
                        row[f"hd95_{s}"] = 5.0
                    writer.writerow(row)


@pytest.fixture
def levels():
    rng = np.random.default_rng(0)
    return {f"patient{i:04d}": float(v) for i, v in enumerate(rng.uniform(0.7, 0.95, 50))}


def test_identical_arms_give_p_one_and_ci_through_zero(tmp_path, levels):
    write_per_sample(tmp_path / "a.csv", levels)
    a = per_patient(tmp_path / "a.csv", "dice_mean")
    out = compare(a, a)
    assert out["n"] == 50
    assert out["p"] == 1.0
    assert out["ci_low"] <= 0.0 <= out["ci_high"]
    assert out["mean_diff"] == 0.0


def test_constant_shift_is_detected(tmp_path, levels):
    write_per_sample(tmp_path / "a.csv", levels, shift=0.02)
    write_per_sample(tmp_path / "b.csv", levels)
    out = compare(per_patient(tmp_path / "a.csv", "dice_lv_myo"),
                  per_patient(tmp_path / "b.csv", "dice_lv_myo"))
    assert out["mean_diff"] == pytest.approx(0.02, abs=0.005)
    assert out["ci_low"] > 0.0
    assert out["p"] < 0.001


def test_per_patient_averages_the_four_images_and_ignores_nan(tmp_path):
    path = tmp_path / "x.csv"
    write_per_sample(path, {"patient0001": 0.8})
    with open(path) as f:
        rows = list(csv.DictReader(f))
    rows[0]["dice_left_atrium"] = "nan"  # a structure absent from GT and prediction
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    values = per_patient(path, "dice_mean")
    assert list(values) == ["patient0001"]
    assert 0.75 < values["patient0001"] < 0.85


def test_holm_adjustment():
    # sorted 0.01, 0.03, 0.04 -> 0.03, 0.06, then 0.04 is lifted to the running 0.06
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]
    assert all(p <= 1.0 for p in holm([0.9, 0.8, 0.7]))


def test_run_family_skips_missing_and_corrects_within_budget(tmp_path, levels):
    root = tmp_path / "outputs"
    write_per_sample(root / ARMS["unet"][20] / "test_per_sample.csv", levels)
    write_per_sample(root / ARMS["medsam2_ft"][20] / "test_per_sample.csv", levels, shift=0.03)
    write_per_sample(root / ARMS["dino3_head"][20] / "test_per_sample.csv", levels, shift=-0.03)
    results = run_family(root, 20)
    tested = {(r["arm_a"], r["arm_b"], r["metric"]) for r in results}
    assert ("medsam2_ft", "unet", "dice_mean") in tested
    assert ("unet", "dino3_head", "dice_lv_myo") in tested
    assert not any(r["arm_a"] == "universeg_k8" for r in results)
    for r in results:
        assert r["p_holm"] >= r["p"]
        assert r["significant_05"]


def write_metrics(path, dice_mean: float):
    path.mkdir(parents=True, exist_ok=True)
    summary = {"dice_mean": dice_mean, "dice_lv_endo": dice_mean + 0.05, "hd95_mean": 6.0,
               "protocol": "text", "by_quality": {}}
    (path / "test_metrics.json").write_text(json.dumps(summary))


def test_aggregate_finds_seed_dirs_and_reports_sd(tmp_path):
    root = tmp_path / "outputs"
    base = ARMS["unet"][20]
    write_metrics(root / base, 0.85)
    write_metrics(root / f"{base}_s1", 0.86)
    write_metrics(root / f"{base}_s2", 0.87)
    write_metrics(root / ARMS["unet"][400], 0.90)
    out = aggregate(root, "unet")
    twenty = out["budgets"]["20"]
    assert twenty["n_seeds"] == 3
    assert twenty["metrics"]["dice_mean"]["mean"] == pytest.approx(0.86)
    assert twenty["metrics"]["dice_mean"]["sd"] == pytest.approx(0.01)
    assert "protocol" not in twenty["metrics"]
    full = out["budgets"]["400"]
    assert full["n_seeds"] == 1 and full["metrics"]["dice_mean"]["sd"] is None
    assert "40" not in out["budgets"]


def test_single_seed_arms_ignore_seed_dirs(tmp_path):
    root = tmp_path / "outputs"
    base = ARMS["seggpt"][20]
    write_metrics(root / base, 0.58)
    write_metrics(root / f"{base}_s1", 0.60)
    out = aggregate(root, "seggpt")
    assert out["budgets"]["20"]["n_seeds"] == 1
