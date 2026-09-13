"""Subset drawing: seed 42 reproduces the committed lists, other seeds give
different lists that are still nested and keep the same stratum quotas."""

from collections import defaultdict
from pathlib import Path

import pytest

from make_subsets import SIZES, draw_subsets, patient_stratum
from medseg_label_efficiency.data.camus import read_split

REPO = Path(__file__).parents[1]


def synthetic_strata() -> dict[str, list[str]]:
    # 400 patients over strata of unequal size, like the real quality mix
    sizes = {"Good|Good": 140, "Good|Medium": 110, "Medium|Medium": 70, "Medium|Poor": 50,
             "Good|Poor": 16, "Poor|Poor": 14}
    strata, n = {}, 0
    for stratum, size in sizes.items():
        strata[stratum] = [f"patient{n + i:04d}" for i in range(size)]
        n += size
    return strata


def stratum_of(strata: dict[str, list[str]]) -> dict[str, str]:
    return {p: s for s, members in strata.items() for p in members}


def test_subsets_nest_and_have_the_requested_sizes():
    subsets = draw_subsets(synthetic_strata(), seed=42)
    assert {k: len(v) for k, v in subsets.items()} == SIZES
    assert set(subsets["p05"]) <= set(subsets["p10"]) <= set(subsets["p25"])


def test_seeds_differ_but_share_the_stratum_quotas():
    strata = synthetic_strata()
    lookup = stratum_of(strata)
    a, b = draw_subsets(strata, seed=1), draw_subsets(strata, seed=2)
    for name in SIZES:
        assert a[name] != b[name]
        assert set(a[name]) <= set(a["p25"]) and set(b[name]) <= set(b["p25"])
        counts_a, counts_b = defaultdict(int), defaultdict(int)
        for p in a[name]:
            counts_a[lookup[p]] += 1
        for p in b[name]:
            counts_b[lookup[p]] += 1
        assert counts_a == counts_b
        assert counts_a["Good|Good"] >= counts_a["Poor|Poor"]


def test_same_seed_is_deterministic():
    strata = synthetic_strata()
    assert draw_subsets(strata, seed=7) == draw_subsets(strata, seed=7)


@pytest.mark.requires_data
@pytest.mark.parametrize("seed, suffix", [(42, ""), (1, "_s1"), (2, "_s2")])
def test_committed_lists_reproduce(data_root, seed, suffix):
    strata = defaultdict(list)
    for patient in read_split(data_root, "train"):
        strata[patient_stratum(data_root, patient)].append(patient)
    subsets = draw_subsets(strata, seed)
    for name, chosen in subsets.items():
        committed = (REPO / "configs" / "subsets" / f"train_{name}{suffix}.txt").read_text()
        assert committed == "\n".join(chosen) + "\n"
