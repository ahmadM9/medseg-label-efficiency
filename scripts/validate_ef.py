"""Validate the ejection-fraction code on ground-truth masks against the Info EF.

    python scripts/validate_ef.py [--data-root data/camus] [--out outputs/ef_validation.json]

Runs Simpson's biplane method (ef.py) on the ground-truth masks of all 500
patients and compares the result with the EF value distributed in each
patient's Info file. The gate decides whether any model-mask EF may be
reported: pass when MAE <= 1.5 EF points and Pearson r >= 0.98; investigate
between 1.5 and 3; fail above 3 or below r 0.95 (a bug in the axis rule).
The Info files carry no volumes, so the volume check is internal: 2CH vs
4CH single-plane volumes must agree, and EDV/ESV must sit in the adult
range. Writes a JSON summary and a Bland-Altman plot.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from monai.data.utils import affine_to_spacing
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from medseg_label_efficiency.data.registry import get_dataset_module  # noqa: E402
from medseg_label_efficiency.ef import ef_from_masks  # noqa: E402

# gate thresholds in EF points, fixed before the first run
PASS_MAE, PASS_R = 1.5, 0.98
FAIL_MAE, FAIL_R = 3.0, 0.95
# adult plausibility range for the end-diastolic volume in ml
EDV_RANGE = (50.0, 250.0)

INK, MUTED, POINT = "#333333", "#767676", "#0072B2"


def load_label_maps(ds, samples: list[dict]) -> tuple[dict, dict]:
    load = Compose([LoadImaged(keys=["label"]), EnsureChannelFirstd(keys=["label"])])
    labels, spacings = {}, {}
    for s in samples:
        data = load({"label": s["label"]})["label"]
        labels[(s["view"], s["phase"])] = data[0].numpy().astype(np.uint8)
        sp = affine_to_spacing(data.affine)[:2]
        spacings[(s["view"], s["phase"])] = (float(sp[0]), float(sp[1]))
    return labels, spacings


def patient_efs(ds, data_root: str, split: str, cavity_id: int, atrium_id: int) -> list[dict]:
    by_patient: dict[str, list[dict]] = {}
    for s in ds.build_samples(data_root, split):
        by_patient.setdefault(s["patient"], []).append(s)
    rows = []
    for patient, samples in by_patient.items():
        labels, spacings = load_label_maps(ds, samples)
        out = ef_from_masks(labels, spacings, cavity_id, atrium_id)
        rows.append({"patient": patient, "split": split, "ef_info": samples[0]["ef"], **out})
    return rows


def agreement(a: np.ndarray, b: np.ndarray) -> dict:
    # a against reference b: error stats plus Bland-Altman limits
    d = a - b
    return {
        "n": int(len(d)),
        "mae": float(np.mean(np.abs(d))),
        "bias": float(np.mean(d)),
        "sd": float(np.std(d, ddof=1)),
        "loa_low": float(np.mean(d) - 1.96 * np.std(d, ddof=1)),
        "loa_high": float(np.mean(d) + 1.96 * np.std(d, ddof=1)),
        "pearson_r": float(np.corrcoef(a, b)[0, 1]),
    }


def gate_tier(stats: dict) -> str:
    if stats["mae"] <= PASS_MAE and stats["pearson_r"] >= PASS_R:
        return "pass"
    if stats["mae"] > FAIL_MAE or stats["pearson_r"] < FAIL_R:
        return "fail"
    return "investigate"


def bland_altman_plot(a: np.ndarray, b: np.ndarray, stats: dict, title: str, path: Path,
                      unit: str = "EF points") -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    mean, diff = (a + b) / 2, a - b
    ax.scatter(mean, diff, s=14, color=POINT, alpha=0.6, edgecolors="none")
    ax.axhline(stats["bias"], color=INK, linewidth=1.2)
    for y in (stats["loa_low"], stats["loa_high"]):
        ax.axhline(y, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(f"bias {stats['bias']:+.2f}", xy=(1.01, stats["bias"]),
                xycoords=("axes fraction", "data"), fontsize=8, color=INK, va="center")
    ax.annotate(f"+1.96 SD {stats['loa_high']:+.2f}", xy=(1.01, stats["loa_high"]),
                xycoords=("axes fraction", "data"), fontsize=8, color=MUTED, va="center")
    ax.annotate(f"-1.96 SD {stats['loa_low']:+.2f}", xy=(1.01, stats["loa_low"]),
                xycoords=("axes fraction", "data"), fontsize=8, color=MUTED, va="center")
    ax.set_xlabel(f"Mean of the two measurements ({unit})", fontsize=9, color=INK)
    ax.set_ylabel(f"Difference ({unit})", fontsize=9, color=INK)
    ax.set_title(title, fontsize=10, color=INK)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/camus")
    parser.add_argument("--dataset", default="camus")
    parser.add_argument("--out", default="outputs/ef_validation.json")
    parser.add_argument("--figures", default="outputs/figures")
    args = parser.parse_args()

    ds = get_dataset_module(args.dataset)
    ids = {name: sid for sid, name in ds.STRUCTURES.items()}
    cavity_id, atrium_id = ids["lv_endo"], ids["left_atrium"]
    rows = []
    for split in ds.SPLITS:
        rows.extend(patient_efs(ds, args.data_root, split, cavity_id, atrium_id))
        print(f"{split}: {len(rows)} patients so far", flush=True)

    ef_ours = np.array([r["ef"] for r in rows])
    ef_info = np.array([r["ef_info"] for r in rows], dtype=float)
    ef_stats = agreement(ef_ours, ef_info)
    tier = gate_tier(ef_stats)

    edv = np.array([r["edv_ml"] for r in rows])
    esv = np.array([r["esv_ml"] for r in rows])
    planes = {}
    for phase, key in (("ED", "edv"), ("ES", "esv")):
        two = np.array([r["single_plane_ml"][phase]["2CH"] for r in rows])
        four = np.array([r["single_plane_ml"][phase]["4CH"] for r in rows])
        planes[key] = agreement(two, four)
    outside = (edv < EDV_RANGE[0]) | (edv > EDV_RANGE[1])
    ranges = {
        "edv_ml": {"min": float(edv.min()), "median": float(np.median(edv)),
                   "max": float(edv.max()), "n_outside_adult_range": int(outside.sum())},
        "esv_ml": {"min": float(esv.min()), "median": float(np.median(esv)),
                   "max": float(esv.max())},
    }
    fallbacks = {}
    for r in rows:
        for rule in r["rules"].values():
            fallbacks[rule] = fallbacks.get(rule, 0) + 1
    worst = sorted(rows, key=lambda r: -abs(r["ef"] - r["ef_info"]))[:10]

    summary = {
        "n_patients": len(rows),
        "gate": {"tier": tier, "pass": {"mae": PASS_MAE, "r": PASS_R},
                 "fail": {"mae": FAIL_MAE, "r": FAIL_R}},
        "ef_vs_info": ef_stats,
        "single_plane_2ch_vs_4ch": planes,
        "volume_ranges": ranges,
        "axis_rules": fallbacks,
        "worst_patients": [
            {"patient": r["patient"], "split": r["split"], "ef": round(r["ef"], 2),
             "ef_info": r["ef_info"], "edv_ml": round(r["edv_ml"], 1),
             "esv_ml": round(r["esv_ml"], 1)}
            for r in worst
        ],
        "per_patient": [
            {"patient": r["patient"], "split": r["split"], "ef": r["ef"], "ef_info": r["ef_info"],
             "edv_ml": r["edv_ml"], "esv_ml": r["esv_ml"], "rules": r["rules"]}
            for r in rows
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    figures = Path(args.figures)
    bland_altman_plot(ef_ours, ef_info, ef_stats, "EF from ground-truth masks vs Info EF",
                      figures / "ef_validation_ba.png")
    two = np.array([r["single_plane_ml"]["ED"]["2CH"] for r in rows])
    four = np.array([r["single_plane_ml"]["ED"]["4CH"] for r in rows])
    bland_altman_plot(two, four, planes["edv"], "EDV single-plane 2CH vs 4CH",
                      figures / "ef_validation_edv_planes_ba.png", unit="ml")

    print(f"\nEF vs Info EF over {len(rows)} patients: MAE {ef_stats['mae']:.2f}, "
          f"bias {ef_stats['bias']:+.2f}, r {ef_stats['pearson_r']:.4f}, "
          f"limits [{ef_stats['loa_low']:+.2f}, {ef_stats['loa_high']:+.2f}]")
    print(f"gate: {tier.upper()}")
    for key, st in planes.items():
        print(f"{key} 2CH vs 4CH: bias {st['bias']:+.1f} ml, limits [{st['loa_low']:+.1f}, "
              f"{st['loa_high']:+.1f}], r {st['pearson_r']:.3f}")
    print(f"EDV {ranges['edv_ml']['min']:.0f} to {ranges['edv_ml']['max']:.0f} ml "
          f"(median {ranges['edv_ml']['median']:.0f}), "
          f"{ranges['edv_ml']['n_outside_adult_range']} outside {EDV_RANGE}; "
          f"ESV {ranges['esv_ml']['min']:.0f} to {ranges['esv_ml']['max']:.0f} ml")
    print(f"axis rules: {fallbacks}")
    print("worst patients:")
    for w in summary["worst_patients"]:
        print(f"  {w['patient']} ({w['split']}): ours {w['ef']:.1f} vs info {w['ef_info']:.0f}, "
              f"EDV {w['edv_ml']} ESV {w['esv_ml']}")
    print(f"wrote {out} and {figures}/ef_validation_*.png")


if __name__ == "__main__":
    main()
