"""Ejection-fraction error of one arm from its saved test masks.

    python scripts/ef_error.py --arm outputs/camus_unet2d [--data-root data/camus]

Reads <arm>/masks/<patient>_<view>_<phase>.npz (written by the evaluation
CLIs with --save-masks), computes Simpson's biplane EF per test patient and
compares it with two references: the EF computed the same way from the
ground-truth masks (the protocol of the CAMUS paper's clinical metrics) and
the EF distributed in the Info files. Reports MAE, bias, Pearson r and
Bland-Altman limits for both, the long-axis fallback count, and writes
<arm>/ef_error.json plus a Bland-Altman plot per reference.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from medseg_label_efficiency.data.registry import get_dataset_module
from medseg_label_efficiency.ef import ef_from_masks
from medseg_label_efficiency.reporting import load_mask
from validate_ef import agreement, bland_altman_plot, load_label_maps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, help="out_dir holding masks/")
    parser.add_argument("--data-root", default="data/camus")
    parser.add_argument("--dataset", default="camus")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    ds = get_dataset_module(args.dataset)
    ids = {name: sid for sid, name in ds.STRUCTURES.items()}
    cavity_id, atrium_id = ids["lv_endo"], ids["left_atrium"]
    arm = Path(args.arm)

    by_patient: dict[str, list[dict]] = {}
    for s in ds.build_samples(args.data_root, args.split):
        by_patient.setdefault(s["patient"], []).append(s)

    rows, failed = [], []
    for patient, samples in by_patient.items():
        gt_labels, spacings = load_label_maps(ds, samples)
        pred_labels = {(s["view"], s["phase"]): load_mask(arm, s) for s in samples}
        gt = ef_from_masks(gt_labels, spacings, cavity_id, atrium_id)
        try:
            pred = ef_from_masks(pred_labels, spacings, cavity_id, atrium_id)
        except ValueError:
            # an empty predicted cavity has no long axis and no volume
            failed.append(patient)
            continue
        rows.append({
            "patient": patient, "ef_pred": pred["ef"], "ef_gt": gt["ef"],
            "ef_info": samples[0]["ef"], "edv_pred": pred["edv_ml"], "esv_pred": pred["esv_ml"],
            "edv_gt": gt["edv_ml"], "esv_gt": gt["esv_ml"], "rules": pred["rules"],
            "n_fallback": pred["n_fallback"],
        })

    pred = np.array([r["ef_pred"] for r in rows])
    ok = np.isfinite(pred)
    refs = {
        "gt_masks": np.array([r["ef_gt"] for r in rows]),
        "info": np.array([r["ef_info"] for r in rows], dtype=float),
    }
    stats = {name: agreement(pred[ok], ref[ok]) for name, ref in refs.items()}
    volumes = {
        "edv": agreement(np.array([r["edv_pred"] for r in rows])[ok],
                         np.array([r["edv_gt"] for r in rows])[ok]),
        "esv": agreement(np.array([r["esv_pred"] for r in rows])[ok],
                         np.array([r["esv_gt"] for r in rows])[ok]),
    }
    rules: dict[str, int] = {}
    for r in rows:
        for rule in r["rules"].values():
            rules[rule] = rules.get(rule, 0) + 1

    summary = {
        "arm": str(arm),
        "split": args.split,
        "n_patients": len(rows),
        "n_failed_empty_cavity": len(failed),
        "failed_patients": failed,
        "n_nonfinite_ef": int((~ok).sum()),
        "ef_vs_gt_masks": stats["gt_masks"],
        "ef_vs_info": stats["info"],
        "volumes_vs_gt_masks_ml": volumes,
        "axis_rules": rules,
        "n_patients_with_fallback": sum(r["n_fallback"] > 0 for r in rows),
        "per_patient": rows,
    }
    (arm / "ef_error.json").write_text(json.dumps(summary, indent=2))
    name = arm.name
    bland_altman_plot(pred[ok], refs["gt_masks"][ok], stats["gt_masks"],
                      f"{name}: EF vs ground-truth-mask EF", arm / "ef_ba_gt.png")
    bland_altman_plot(pred[ok], refs["info"][ok], stats["info"],
                      f"{name}: EF vs Info EF", arm / "ef_ba_info.png")

    for ref, st in stats.items():
        print(f"{name} EF vs {ref}: n {st['n']}, MAE {st['mae']:.2f}, bias {st['bias']:+.2f}, "
              f"r {st['pearson_r']:.3f}, limits [{st['loa_low']:+.2f}, {st['loa_high']:+.2f}]")
    for key, st in volumes.items():
        print(f"{key} vs GT: MAE {st['mae']:.1f} ml, bias {st['bias']:+.1f}, "
              f"r {st['pearson_r']:.3f}")
    print(f"axis rules {rules}; patients with a fallback: "
          f"{summary['n_patients_with_fallback']}; empty cavities: {len(failed)}")
    print(f"wrote {arm}/ef_error.json and ef_ba_*.png")


if __name__ == "__main__":
    main()
