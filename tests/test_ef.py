"""Simpson's biplane volumes and the long-axis rule on synthetic shapes."""

import numpy as np
import pytest
from scipy.ndimage import rotate

from medseg_label_efficiency.ef import biplane_volume, ef_from_masks, long_axis, single_plane_volume

A_MM, B_MM = 35.0, 20.0  # long and short semi-axis of the synthetic cavity
SPACING = (0.3, 0.3)


def hemi_ellipsoid(spacing=SPACING, gap_px: int = 0, with_atrium: bool = True, full=False):
    # a half ellipse cut flat at its widest point (the base), apex along +row;
    # an atrium block sits against the flat face. its 3D volume of revolution
    # is 2/3 pi a b^2, the analytic reference for the disc method. full=True
    # keeps the whole ellipse (4/3 pi a b^2) for the no-atrium fallback
    sr, sc = spacing
    h, w = int(3 * A_MM / sr) + 60, int(2 * B_MM / sc) + 60
    rr, cc = np.mgrid[:h, :w]
    r0, c0 = 40 + int(A_MM / sr), w // 2
    y = (rr - r0) * sr
    x = (cc - c0) * sc
    cavity = (y / A_MM) ** 2 + (x / B_MM) ** 2 <= 1
    if not full:
        cavity &= y >= 0
    labels = np.zeros((h, w), dtype=np.uint8)
    labels[cavity] = 1
    if with_atrium:
        top = r0 - gap_px
        labels[max(top - 25, 0) : top, c0 - int(B_MM / sc) : c0 + int(B_MM / sc)] = 3
    return labels, (r0, c0)


def analytic_ml() -> float:
    return 2 / 3 * np.pi * A_MM * B_MM**2 / 1000


def test_single_plane_volume_matches_the_analytic_ellipsoid():
    labels, _ = hemi_ellipsoid()
    out = single_plane_volume(labels == 1, labels == 3, SPACING)
    assert out["rule"] == "annulus"
    assert out["volume_ml"] == pytest.approx(analytic_ml(), rel=0.03)
    assert out["length_mm"] == pytest.approx(A_MM, rel=0.02)


def test_anisotropic_spacing_gives_the_same_volume():
    labels, _ = hemi_ellipsoid(spacing=(0.5, 0.25))
    out = single_plane_volume(labels == 1, labels == 3, (0.5, 0.25))
    assert out["volume_ml"] == pytest.approx(analytic_ml(), rel=0.03)


def test_rotation_does_not_change_the_volume():
    labels, _ = hemi_ellipsoid()
    reference = single_plane_volume(labels == 1, labels == 3, SPACING)["volume_ml"]
    for angle in (25, 90, 137):
        rotated = rotate(labels, angle, order=0, reshape=True, prefilter=False)
        out = single_plane_volume(rotated == 1, rotated == 3, SPACING)
        assert out["rule"] == "annulus"
        assert out["volume_ml"] == pytest.approx(reference, rel=0.02)


def test_annulus_midpoint_lands_on_the_flat_face():
    labels, (r0, c0) = hemi_ellipsoid()
    axis = long_axis(labels == 1, labels == 3, SPACING)
    base_px = axis["base"] / np.array(SPACING)
    assert abs(base_px[0] - r0) <= 1.5 and abs(base_px[1] - c0) <= 1.5
    apex_px = axis["apex"] / np.array(SPACING)
    assert apex_px[0] == pytest.approx(r0 + A_MM / SPACING[0], abs=2)


def test_fallback_rules_are_flagged():
    gapped, (r0, c0) = hemi_ellipsoid(gap_px=3)
    axis = long_axis(gapped == 1, gapped == 3, SPACING)
    assert axis["rule"] == "nearest"
    base_px = axis["base"] / np.array(SPACING)
    assert abs(base_px[0] - r0) <= 1.5 and abs(base_px[1] - c0) <= 1.5

    # no atrium at all: the principal axis of a whole ellipse is its long axis
    alone, _ = hemi_ellipsoid(with_atrium=False, full=True)
    out = single_plane_volume(alone == 1, alone == 3, SPACING)
    assert out["rule"] == "pca"
    assert out["volume_ml"] == pytest.approx(2 * analytic_ml(), rel=0.03)
    assert out["length_mm"] == pytest.approx(2 * A_MM, rel=0.02)


def test_biplane_equals_single_plane_for_identical_views():
    labels, _ = hemi_ellipsoid()
    cav, atr = labels == 1, labels == 3
    out = biplane_volume(cav, atr, SPACING, cav, atr, SPACING)
    single = single_plane_volume(cav, atr, SPACING)["volume_ml"]
    assert out["volume_ml"] == pytest.approx(single, rel=1e-6)
    assert out["rules"] == {"2CH": "annulus", "4CH": "annulus"}


def test_ef_from_masks_reports_the_expected_fraction():
    ed, _ = hemi_ellipsoid()
    es = ed.copy()
    # ES: shrink the cavity to the inner half of the short axis
    rr, cc = np.mgrid[: ed.shape[0], : ed.shape[1]]
    c0 = ed.shape[1] // 2
    es[(ed == 1) & (np.abs(cc - c0) > B_MM / SPACING[1] / 2)] = 0
    labels = {("2CH", "ED"): ed, ("4CH", "ED"): ed, ("2CH", "ES"): es, ("4CH", "ES"): es}
    spacings = dict.fromkeys(labels, SPACING)
    out = ef_from_masks(labels, spacings, cavity_id=1, atrium_id=3)
    assert out["edv_ml"] == pytest.approx(analytic_ml(), rel=0.03)
    assert 0 < out["esv_ml"] < out["edv_ml"]
    assert out["ef"] == pytest.approx(100 * (1 - out["esv_ml"] / out["edv_ml"]))
    assert out["n_fallback"] == 0
    assert set(out["rules"]) == {"2CH_ED", "4CH_ED", "2CH_ES", "4CH_ES"}
