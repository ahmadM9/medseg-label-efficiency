"""Render the label-efficiency figures from the evaluation results.

    python scripts/make_figures.py [--outputs outputs] [--out outputs/figures]

Reads whatever results exist (missing points are skipped, so this works
before all runs are done): one curve per trained or in-context arm over the
four label budgets, and the four tiny-model zero-shot configurations as
horizontal reference lines. When scripts/aggregate_seeds.py has written
outputs/seeds/<arm>.json, a curve shows the mean over seeds with a band of
one SD; arms with a single seed are drawn without a band and say so in the
legend.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aggregate_seeds import ARMS, ZERO_SHOT  # noqa: E402
from medseg_label_efficiency.data.registry import get_dataset_module  # noqa: E402

# (arm, legend label, colour, marker, line style). okabe-ito for the first
# four, validated colourblind-safe; the two in-context arms passed the
# normal-vision check against them and are told apart by marker and the
# dotted line as well (no colour-only identity). identity is fixed, never
# cycled: a colour names one entity in every figure
CURVES = [
    ("unet", "U-Net (supervised)", "#0072B2", "o", "-"),
    ("medsam2_ft", "MedSAM2 (fine-tuned)", "#009E73", "s", "-"),
    ("dino3_head", "DINOv3 frozen + head", "#56B4E9", "^", "-"),
    ("dino2_head", "DINOv2 frozen + head", "#000000", "v", "-"),
    ("universeg", "UniverSeg in-context (K=32)", "#A6611A", "D", ":"),
    ("seggpt", "SegGPT in-context (K=8)", "#7B3294", "P", ":"),
    # same strategy as the U-Net (train from scratch), so the same hue, told
    # apart by the dashed line and marker; the palette checker found no free
    # hue against the six above
    ("nnunet", "nnU-Net (supervised)", "#0072B2", "x", "--"),
]
# the zero-shot references: the same green as the fine-tuned MedSAM2 curve
# because it is the same entity at zero labels
ZERO_SHOT_LINES = [
    ("SAM 2.1 · box", "sam2_box", "#E69F00"),
    ("MedSAM2 · box", "medsam2_box", "#009E73"),
    ("MedSAM2 · point", "medsam2_point", "#CC79A7"),
    ("SAM 2.1 · point", "sam2_point", "#D55E00"),
]
INK, MUTED = "#333333", "#767676"

DATASET = get_dataset_module("camus")
STRUCTURES = list(DATASET.STRUCTURES.values())
TITLES = DATASET.STRUCTURE_TITLES


def load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"skipping missing {path}")
        return None
    return json.loads(path.read_text())


def load_curve(root: Path, arm: str, metric: str) -> list[tuple[int, float, float | None, int]]:
    # (budget, mean, sd or None, n_seeds) per budget; the seed aggregate is
    # preferred, the seed-0 result file is the fallback before aggregation
    seeds = load_json(root / "seeds" / f"{arm}.json") if (root / "seeds").exists() else None
    points = []
    for budget, out_dir in ARMS[arm].items():
        entry = (seeds or {}).get("budgets", {}).get(str(budget))
        if entry is not None and metric in entry["metrics"]:
            m = entry["metrics"][metric]
            points.append((budget, m["mean"], m["sd"], entry["n_seeds"]))
            continue
        summary = load_json(root / out_dir / "test_metrics.json")
        if summary is not None:
            points.append((budget, summary[metric], None, 1))
    return points


def draw(ax, root: Path, metric: str) -> None:
    lines = []
    for label, key, color in ZERO_SHOT_LINES:
        summary = load_json(root / ZERO_SHOT[key] / "test_metrics.json")
        if summary is not None:
            lines.append((summary[metric], label, color))
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

    # curves converge at the largest budget, so values live in the tables;
    # only the low-budget end (where the arms differ) is labeled here
    for arm, name, color, marker, style in CURVES:
        points = load_curve(root, arm, metric)
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        banded = [(x, y, sd) for x, y, sd, n in points if sd is not None and n > 1]
        if banded:
            bx, by, bsd = zip(*banded, strict=True)
            ax.fill_between(
                bx, [y - s for y, s in zip(by, bsd, strict=True)],
                [y + s for y, s in zip(by, bsd, strict=True)],
                color=color, alpha=0.15, linewidth=0,
            )
        else:
            name = f"{name}, single seed"
        ax.plot(
            xs, ys, color=color, marker=marker, markersize=6, linewidth=2, label=name,
            linestyle=style,
        )
        ax.annotate(
            f"{ys[0]:.2f}", xy=(xs[0], ys[0]), xytext=(-4, 6),
            textcoords="offset points", fontsize=8, color=INK, ha="right",
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
    ax.legend(
        loc="lower right", fontsize=8, frameon=False, labelcolor=INK,
        handlelength=1.6, borderaxespad=1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--out", default="outputs/figures")
    args = parser.parse_args()
    root = Path(args.outputs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    draw(ax, root, "dice_mean")
    ax.set_ylabel("Test Dice (mean over structures, excl. background)", fontsize=9, color=INK)
    ax.set_xlabel("Labeled training patients", fontsize=9, color=INK)
    ax.set_title(
        "Train, adapt, or prompt? Segmentation quality vs annotation budget",
        fontsize=11, color=INK, pad=12,
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    fig.savefig(out_dir / "label_efficiency.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, structure in zip(axes, STRUCTURES, strict=True):
        draw(ax, root, f"dice_{structure}")
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
