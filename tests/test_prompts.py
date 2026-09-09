"""Prompt-simulation tests, including the myocardium-ring (donut) case."""

import numpy as np

from medseg_label_efficiency.prompts import box_from_mask, point_from_mask, sample_rng


def _donut(size=100, center=50, outer=30, inner=20):
    yy, xx = np.mgrid[:size, :size]
    dist = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    return (dist <= outer) & (dist >= inner)


def test_interior_point_lands_on_donut_ring():
    ring = _donut()
    # center of mass falls in the hole — the classic failure this code avoids
    com = tuple(int(v) for v in np.round(np.argwhere(ring).mean(axis=0)))
    assert not ring[com]
    point = point_from_mask(ring)
    assert ring[point]


def test_interior_point_is_deep_inside_solid_shape():
    mask = np.zeros((60, 60), dtype=bool)
    mask[20:40, 10:50] = True
    r, c = point_from_mask(mask)
    assert mask[r, c]
    assert 25 <= r <= 34  # well away from the 20/40 edges


def test_box_without_jitter_is_tight():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 15:30] = True
    assert box_from_mask(mask, jitter_px=0) == (10, 15, 19, 29)


def test_box_jitter_reproducible_and_clipped():
    mask = np.zeros((50, 50), dtype=bool)
    mask[0:10, 40:50] = True  # touches two image borders
    a = box_from_mask(mask, jitter_px=5, rng=sample_rng("patient0001", 1))
    b = box_from_mask(mask, jitter_px=5, rng=sample_rng("patient0001", 1))
    assert a == b
    r0, c0, r1, c1 = a
    assert 0 <= r0 <= r1 <= 49 and 0 <= c0 <= c1 <= 49


def test_empty_mask_returns_none():
    empty = np.zeros((30, 30), dtype=bool)
    assert box_from_mask(empty) is None
    assert point_from_mask(empty) is None
