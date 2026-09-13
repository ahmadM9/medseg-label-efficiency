"""Generate the nested, quality-stratified label subsets.

    python scripts/make_subsets.py [--data-root data/camus] [--seed 42] [--suffix ""]

From the official 400 training patients, draws 20 / 40 / 100 (5 / 10 / 25%)
such that each subset keeps the dataset's image-quality mix and each smaller
subset is contained in the next larger one (a growing annotation budget).
Writes configs/subsets/train_p{05,10,25}<suffix>.txt, committed to the repo
so the experiment is exactly reproducible. Seed 42 with no suffix is the
seed-0 family every result so far used; --seed 1 --suffix _s1 and
--seed 2 --suffix _s2 are the extra subset draws behind the error bars.

Method: stratum = the sorted pair of per-view ImageQuality labels; one seeded
shuffle per stratum; per-stratum quotas proportional to stratum size
(largest-remainder rounding, forced monotone across subset sizes); every
subset takes prefixes of the same shuffled order, which guarantees nesting.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from medseg_label_efficiency.data.camus import parse_info, read_split

SEED = 42
SIZES = {"p05": 20, "p10": 40, "p25": 100}


def patient_stratum(data_root: Path, patient: str) -> str:
    qualities = sorted(
        parse_info(data_root / "database_nifti" / patient / f"Info_{view}.cfg")
        .get("ImageQuality", "Unknown")
        for view in ("2CH", "4CH")
    )
    return "|".join(qualities)


def quotas_for(sizes: list[int], stratum_sizes: dict[str, int], total: int) -> dict:
    # largest-remainder rounding, and a stratum's quota may never shrink as
    # the subset grows, otherwise prefixes of one shuffle would not nest
    quotas = {name: {} for name in stratum_sizes}
    previous = dict.fromkeys(stratum_sizes, 0)
    for k in sizes:
        ideal = {s: k * n / total for s, n in stratum_sizes.items()}
        floor = {s: max(int(v), previous[s]) for s, v in ideal.items()}
        remainder = k - sum(floor.values())
        by_frac = sorted(ideal, key=lambda s: ideal[s] - int(ideal[s]), reverse=True)
        for s in by_frac:
            if remainder <= 0:
                break
            if floor[s] < stratum_sizes[s]:
                floor[s] += 1
                remainder -= 1
        assert sum(floor.values()) == k, "quota rounding failed"
        for s in stratum_sizes:
            quotas[s][k] = floor[s]
        previous = {s: floor[s] for s in stratum_sizes}
    return quotas


def draw_subsets(strata: dict[str, list[str]], seed: int) -> dict[str, list[str]]:
    # strata: stratum name -> its patients; returns subset name -> sorted patients
    strata = {s: sorted(m) for s, m in strata.items()}
    rng = np.random.default_rng(seed)
    for members in strata.values():
        rng.shuffle(members)
    total = sum(len(m) for m in strata.values())
    stratum_sizes = {s: len(m) for s, m in strata.items()}
    quotas = quotas_for(sorted(SIZES.values()), stratum_sizes, total)

    subsets = {}
    chosen_prev: set[str] = set()
    for name, k in sorted(SIZES.items(), key=lambda kv: kv[1]):
        chosen = []
        for s, members in strata.items():
            chosen.extend(members[: quotas[s][k]])
        assert chosen_prev <= set(chosen), "nesting violated"
        chosen_prev = set(chosen)
        subsets[name] = sorted(chosen)
    return subsets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/camus")
    parser.add_argument("--out", default="configs/subsets")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--suffix", default="", help="appended to the file stem, e.g. _s1")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    strata = defaultdict(list)
    for patient in read_split(data_root, "train"):
        strata[patient_stratum(data_root, patient)].append(patient)
    subsets = draw_subsets(strata, args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, chosen in subsets.items():
        path = out_dir / f"train_{name}{args.suffix}.txt"
        path.write_text("\n".join(chosen) + "\n")
        counts = defaultdict(int)
        for p in chosen:
            counts[patient_stratum(data_root, p)] += 1
        dist = ", ".join(f"{s}: {n}" for s, n in sorted(counts.items()))
        print(f"{path}  ({len(chosen)} patients, seed {args.seed})  {dist}")


if __name__ == "__main__":
    main()
