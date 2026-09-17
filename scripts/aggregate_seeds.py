"""Collect the per-seed test metrics of every arm into mean and SD per budget.

    python scripts/aggregate_seeds.py [--outputs outputs]

For each arm and budget the seed-0 run lives in the committed out_dir and the
extra seeds in <out_dir>_s1 and <out_dir>_s2 (the seed runs are named
so). Writes outputs/seeds/<arm>.json with, per budget, the number of
seeds found and mean, SD (sample SD, null below two seeds) and the values of
every dice/hd95/overlap metric. make_figures.py draws the bands from it.
"""

import argparse
import json
from pathlib import Path

import numpy as np

BUDGETS = (20, 40, 100, 400)
SUFFIX = {20: "p05", 40: "p10", 100: "p25", 400: "full"}

# arm -> budget -> seed-0 out_dir relative to the outputs root. the U-Net's
# full-budget run predates the suffix convention, hence its bare name
ARMS = {
    "unet": {20: "camus_unet2d_p05", 40: "camus_unet2d_p10", 100: "camus_unet2d_p25",
             400: "camus_unet2d"},
    "medsam2_ft": {b: f"medsam2_ft_{SUFFIX[b]}" for b in BUDGETS},
    "dino3_head": {b: f"dino3_head_{SUFFIX[b]}" for b in BUDGETS},
    "dino2_head": {b: f"dino2_head_{SUFFIX[b]}" for b in BUDGETS},
    "universeg": {b: f"universeg_{SUFFIX[b]}" for b in BUDGETS},
    "universeg_k8": {b: f"universeg_{SUFFIX[b]}_k8" for b in BUDGETS},
    "universeg_k64": {b: f"universeg_{SUFFIX[b]}_k64" for b in (40, 100, 400)},
    "seggpt": {b: f"seggpt_{SUFFIX[b]}" for b in BUDGETS},
    # automatic-prompt cascade: MedSAM2 behind boxes drawn from another arm's
    # test predictions; the seed-0 U-Net rows are the box-drawer check, the
    # nnU-Net rows (_nn) the reported version, since nnU-Net won at 20 patients
    "medsam2_cascade": {b: f"medsam2_cascade_{SUFFIX[b]}" for b in BUDGETS},
    "medsam2_ft_cascade": {b: f"medsam2_ft_cascade_{SUFFIX[b]}" for b in BUDGETS},
    "medsam2_cascade_nn": {b: f"medsam2_cascade_nn_{SUFFIX[b]}" for b in BUDGETS},
    "medsam2_ft_cascade_nn": {b: f"medsam2_ft_cascade_nn_{SUFFIX[b]}" for b in BUDGETS},
    "sam2_large_ft": {20: "sam2_large_ft_p05", 40: "sam2_large_ft_p10"},
    # nnU-Net v2 with its defaults (single seed by decision); the no-mirroring
    # inference of the same checkpoints is the footnote check
    "nnunet": {b: f"nnunet_{SUFFIX[b]}" for b in BUDGETS},
    "nnunet_nomirror": {b: f"nnunet_nomirror_{SUFFIX[b]}" for b in BUDGETS},
}
# arms that were run with seeds 1 and 2; the rest is single seed by decision
SEEDED = ("unet", "medsam2_ft", "dino3_head", "universeg")
EXTRA_SEEDS = (1, 2)
# zero-shot rows have no budget; regenerated with per-image seeds in Stage C
ZERO_SHOT = {
    "sam2_box": "sam2_box", "sam2_point": "sam2_point",
    "medsam2_box": "medsam2_box", "medsam2_point": "medsam2_point",
    "sam2_bplus_box": "sam2_bplus_box", "sam2_bplus_point": "sam2_bplus_point",
    "sam2_large_box": "sam2_large_box", "sam2_large_point": "sam2_large_point",
}


def seed_dirs(root: Path, arm: str, budget: int) -> list[Path]:
    base = root / ARMS[arm][budget]
    dirs = [base]
    if arm in SEEDED:
        dirs += [Path(f"{base}_s{s}") for s in EXTRA_SEEDS]
    return [d for d in dirs if (d / "test_metrics.json").exists()]


def aggregate(root: Path, arm: str) -> dict:
    out = {"arm": arm, "budgets": {}}
    for budget in ARMS[arm]:
        dirs = seed_dirs(root, arm, budget)
        if not dirs:
            continue
        summaries = [json.loads((d / "test_metrics.json").read_text()) for d in dirs]
        keys = [k for k, v in summaries[0].items()
                if isinstance(v, int | float) and k.startswith(("dice_", "hd95_", "overlap_"))]
        metrics = {}
        for k in keys:
            values = [s[k] for s in summaries if k in s]
            metrics[k] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "values": values,
            }
        out["budgets"][str(budget)] = {
            "n_seeds": len(dirs), "dirs": [str(d) for d in dirs], "metrics": metrics,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()
    root = Path(args.outputs)
    seeds_dir = root / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        result = aggregate(root, arm)
        if not result["budgets"]:
            print(f"{arm}: no results yet")
            continue
        (seeds_dir / f"{arm}.json").write_text(json.dumps(result, indent=2))
        cells = []
        for budget, entry in result["budgets"].items():
            m = entry["metrics"]["dice_mean"]
            sd = f" +/- {m['sd']:.3f}" if m["sd"] is not None else ""
            cells.append(f"{budget}: {m['mean']:.3f}{sd} (n={entry['n_seeds']})")
        print(f"{arm}: " + "; ".join(cells))
    print(f"wrote {seeds_dir}/<arm>.json")


if __name__ == "__main__":
    main()
