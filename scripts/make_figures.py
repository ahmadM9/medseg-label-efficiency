"""Render the label-efficiency figures from the evaluation JSONs.

    python scripts/make_figures.py [--out outputs/figures]

Reads whatever test_metrics.json files exist (missing points are skipped, so
this works before all runs are done): the supervised U-Net at each label
budget, and the four zero-shot configurations as horizontal reference lines.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# x = number of labeled training patients
SUPERVISED = [
    (20, "outputs/camus_unet2d_p05/test_metrics.json"),
    (40, "outputs/camus_unet2d_p10/test_metrics.json"),
    (100, "outputs/camus_unet2d_p25/test_metrics.json"),
    (400, "outputs/camus_unet2d/test_metrics.json"),
]
ZERO_SHOT = [
    ("SAM 2.1 · box", "outputs/kaggle-sam-eval/outputs/sam2_box/test_metrics.json"),
    ("MedSAM2 · box", "outputs/kaggle-sam-eval/outputs/medsam2_box/test_metrics.json"),
    ("MedSAM2 · point", "outputs/kaggle-sam-eval/outputs/medsam2_point/test_metrics.json"),
    ("SAM 2.1 · point", "outputs/kaggle-sam-eval/outputs/sam2_point/test_metrics.json"),
]
# Okabe-Ito, validated colorblind-safe; identity is fixed, never cycled
COLOR_SUPERVISED = "#0072B2"
COLORS_ZERO_SHOT = ["#E69F00", "#009E73", "#CC79A7", "#D55E00"]
INK, MUTED = "#333333", "#767676"
STRUCTURES = ["lv_endo", "lv_myo", "left_atrium"]
TITLES = {"lv_endo": "LV endocardium", "lv_myo": "LV myocardium", "left_atrium": "Left atrium"}


def load(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        print(f"skipping missing {p}")
        return None
    with open(p) as f:
        return json.load(f)


def draw(ax, metric_key: str) -> None:
    points = [(n, s[metric_key]) for n, path in SUPERVISED if (s := load(path))]
    lines = []
    for (label, path), color in zip(ZERO_SHOT, COLORS_ZERO_SHOT, strict=True):
        summary = load(path)
        if summary is not None:
            lines.append((summary[metric_key], label, color))
    # dashed line at the true value; label text dodged apart when lines coincide
    label_ys = [y for y, _, _ in lines]
    order = sorted(range(len(lines)), key=lambda i: label_ys[i])
    min_gap = 0.03
    for a, b in zip(order, order[1:], strict=False):
        if label_ys[b] - label_ys[a] < min_gap:
            label_ys[b] = label_ys[a] + min_gap
    for (y, label, color), label_y in zip(lines, label_ys, strict=True):
        ax.axhline(y, color=color, linestyle="--", linewidth=1.4)
        ax.annotate(
            f"{label}  {y:.2f}", xy=(1.01, label_y), xycoords=("axes fraction", "data"),
            fontsize=8, color=color, va="center",
        )
    if points:
        xs, ys = zip(*points, strict=True)
        ax.plot(xs, ys, color=COLOR_SUPERVISED, marker="o", markersize=6, linewidth=2)
        for x, y in points:
            ax.annotate(
                f"{y:.2f}", xy=(x, y), xytext=(0, 9), textcoords="offset points",
                fontsize=8, color=INK, ha="center",
            )
        ax.annotate(
            "U-Net (supervised)", xy=(xs[0], ys[0]), xytext=(0, -16),
            textcoords="offset points", fontsize=9, color=COLOR_SUPERVISED, ha="left",
        )
    ax.set_xscale("log")
    ax.set_xticks([20, 40, 100, 400])
    ax.set_xticklabels(["20\n(5%)", "40\n(10%)", "100\n(25%)", "400\n(100%)"], fontsize=8)
    ax.minorticks_off()
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/figures")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    draw(ax, "dice_mean")
    ax.set_ylabel("Test Dice (mean over structures, excl. background)", fontsize=9, color=INK)
    ax.set_xlabel("Labeled training patients", fontsize=9, color=INK)
    ax.set_title(
        "How many labels until supervised beats zero-shot?", fontsize=11, color=INK, pad=12
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    fig.savefig(out_dir / "label_efficiency.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, structure in zip(axes, STRUCTURES, strict=True):
        draw(ax, f"dice_{structure}")
        ax.set_title(TITLES[structure], fontsize=10, color=INK)
    axes[0].set_ylabel("Test Dice", fontsize=9, color=INK)
    axes[1].set_xlabel("Labeled training patients", fontsize=9, color=INK)
    fig.tight_layout()
    fig.subplots_adjust(right=0.9)
    fig.savefig(out_dir / "label_efficiency_structures.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out_dir}/label_efficiency.png and label_efficiency_structures.png")


if __name__ == "__main__":
    main()
