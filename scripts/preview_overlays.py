"""Render a few CAMUS image/mask overlays to PNG for visual inspection.

    python scripts/preview_overlays.py [--data-root data/camus] [--n 8]

Picks samples spread across views, cardiac phases and image qualities so a
quick look verifies that masks line up with the anatomy in easy and hard cases.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from medseg_label_efficiency.data.camus import build_samples, get_transforms

OVERLAY_COLORS = ListedColormap(["none", "tab:red", "tab:blue", "tab:green"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/camus")
    parser.add_argument("--out", default="outputs/previews")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    samples = build_samples(args.data_root, "train")
    # spread selection across quality/view/phase combinations
    samples.sort(key=lambda s: (s["quality"], s["view"], s["phase"], s["patient"]))
    picks = samples[:: max(1, len(samples) // args.n)][: args.n]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = get_transforms(train=False)

    for sample in picks:
        data = transform(dict(sample))
        image = data["image"][0].numpy()
        label = data["label"][0].numpy()

        fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
        for ax in axes:
            ax.imshow(np.rot90(image), cmap="gray")
            ax.axis("off")
        axes[1].imshow(
            np.rot90(np.ma.masked_equal(label, 0)),
            cmap=OVERLAY_COLORS, alpha=0.4, vmin=0, vmax=3,
        )
        axes[0].set_title("image")
        axes[1].set_title("endo (red) / myo (blue) / LA (green)")
        name = f"{sample['patient']}_{sample['view']}_{sample['phase']}_{sample['quality']}"
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=120)
        plt.close(fig)
        print(f"wrote {out_dir / name}.png")


if __name__ == "__main__":
    main()
