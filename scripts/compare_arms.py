"""Paired per-patient significance tests for the pre-declared arm comparisons.

    python scripts/compare_arms.py [--outputs outputs]

The unit is the patient: each test patient's four images are averaged, so
the 50 test patients give 50 paired differences per comparison. Each
comparison reports the Wilcoxon signed-rank p-value and a 95% bootstrap
confidence interval of the mean difference (10,000 resamples, fixed seed),
with Holm correction over the family of comparisons declared for that
budget in FAMILY. Nothing outside FAMILY is tested. Writes
outputs/stats/<budget>.csv and outputs/stats/comparisons.md.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from aggregate_seeds import ARMS, ZERO_SHOT

N_BOOT = 10_000
BOOT_SEED = 0

# the pre-declared family per budget: (arm A, arm B, metric); A minus B is
# the reported difference. mean Dice and myocardium Dice for every pair,
# left atrium only for zero-shot MedSAM2 vs U-Net at the two small budgets
PAIRS = [
    ("medsam2_ft", "unet"),
    ("medsam2_ft_cascade", "unet"),
    ("unet", "dino3_head"),
    ("dino3_head", "dino2_head"),
    ("unet", "universeg"),
    ("universeg_k8", "seggpt"),
    # pre-declared 2026-09-15, before any nnU-Net run
    ("nnunet", "unet"),
    ("medsam2_ft", "nnunet"),
    # pre-declared 2026-09-17, before the nnU-Net-box cascade ran: the fair
    # automatic pipeline against the model that draws its boxes
    ("medsam2_ft_cascade_nn", "nnunet"),
]
FAMILY = {
    budget: [(a, b, m) for a, b in PAIRS for m in ("dice_mean", "dice_lv_myo")]
    for budget in (20, 40, 100, 400)
}
for _budget in (20, 40):
    FAMILY[_budget].append(("medsam2_box", "unet", "dice_left_atrium"))


def arm_dir(root: Path, arm: str, budget: int) -> Path:
    if arm in ZERO_SHOT:
        return root / ZERO_SHOT[arm]
    return root / ARMS[arm][budget]


def per_patient(csv_path: Path, metric: str) -> dict[str, float]:
    # one value per patient: the mean over its images of the metric, where
    # dice_mean is the mean over structures within an image (nan-aware, so a
    # structure absent from both prediction and GT does not count)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    dice_cols = [c for c in rows[0] if c.startswith("dice_") and c != "dice_mean"]
    values: dict[str, list[float]] = {}
    for row in rows:
        if metric == "dice_mean":
            v = float(np.nanmean([float(row[c]) for c in dice_cols]))
        else:
            v = float(row[metric])
        values.setdefault(row["patient"], []).append(v)
    return {p: float(np.nanmean(v)) for p, v in values.items()}


def compare(a: dict[str, float], b: dict[str, float]) -> dict:
    patients = sorted(set(a) & set(b))
    d = np.array([a[p] - b[p] for p in patients])
    d = d[np.isfinite(d)]
    if np.all(d == 0):
        p_value = 1.0  # identical arms: nothing to rank
    else:
        p_value = float(wilcoxon(d).pvalue)
    rng = np.random.default_rng(BOOT_SEED)
    boots = rng.choice(d, size=(N_BOOT, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": int(len(d)),
        "mean_diff": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p": p_value,
    }


def holm(p_values: list[float]) -> list[float]:
    # step-down Holm: adjusted p_i = max over j <= i of min(1, (m - j + 1) p_(j))
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def run_family(root: Path, budget: int) -> list[dict]:
    results = []
    for arm_a, arm_b, metric in FAMILY[budget]:
        paths = [arm_dir(root, arm, budget) / "test_per_sample.csv" for arm in (arm_a, arm_b)]
        if not all(p.exists() for p in paths):
            print(f"skipping {arm_a} vs {arm_b} ({metric}) at {budget}: missing results")
            continue
        stats = compare(per_patient(paths[0], metric), per_patient(paths[1], metric))
        results.append({"budget": budget, "arm_a": arm_a, "arm_b": arm_b, "metric": metric,
                        **stats})
    if results:
        for row, p_adj in zip(results, holm([r["p"] for r in results]), strict=True):
            row["p_holm"] = p_adj
            row["significant_05"] = p_adj < 0.05
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()
    root = Path(args.outputs)
    stats_dir = root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    fields = ["budget", "arm_a", "arm_b", "metric", "n", "mean_diff", "ci_low", "ci_high", "p",
              "p_holm", "significant_05"]
    lines = ["| budget | A | B | metric | n | A - B | 95% CI | p | p (Holm) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for budget in sorted(FAMILY):
        results = run_family(root, budget)
        if not results:
            continue
        with open(stats_dir / f"{budget}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        for r in results:
            mark = "*" if r["significant_05"] else ""
            lines.append(
                f"| {budget} | {r['arm_a']} | {r['arm_b']} | {r['metric']} | {r['n']} | "
                f"{r['mean_diff']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
                f"{r['p']:.4f} | {r['p_holm']:.4f}{mark} |"
            )
    (stats_dir / "comparisons.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {stats_dir}/<budget>.csv and comparisons.md")


if __name__ == "__main__":
    main()
