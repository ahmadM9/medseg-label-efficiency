import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

# left-ventricular volumes by Simpson's biplane method of discs, computed
# from the cavity mask of the 2CH and 4CH views at ED and ES, and the
# ejection fraction from them. everything is in millimetres from the NIfTI
# spacing, so anisotropic pixels are handled by construction

N_DISCS = 20
NEAREST_TOL_MM = 1.0


def _mm(points: np.ndarray, spacing) -> np.ndarray:
    return points * np.asarray(spacing, dtype=float)


def _boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask, border_value=0)


def _hinge_midpoint(points_mm: np.ndarray) -> np.ndarray:
    # the two ends of the contact segment are the annulus hinge points; the
    # base of the long axis is their midpoint, as in the clinical method
    if len(points_mm) == 1:
        return points_mm[0]
    d = np.linalg.norm(points_mm[:, None] - points_mm[None], axis=-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return (points_mm[i] + points_mm[j]) / 2


def long_axis(cavity: np.ndarray, atrium: np.ndarray, spacing) -> dict:
    # base = midpoint of the segment where the cavity touches the atrium (the
    # mitral annulus), apex = the cavity pixel farthest from the base. with no
    # contact the nearest cavity boundary segment stands in; with no atrium at
    # all the principal axis of the cavity does. the rule used is returned so
    # fallbacks can be counted per arm
    cavity = np.asarray(cavity, dtype=bool)
    atrium = np.asarray(atrium, dtype=bool)
    if not cavity.any():
        raise ValueError("empty cavity mask")
    pixels = _mm(np.argwhere(cavity), spacing)
    edge = _boundary(cavity)
    contact = edge & binary_dilation(atrium, structure=np.ones((3, 3), dtype=bool))
    if contact.any():
        rule = "annulus"
        base = _hinge_midpoint(_mm(np.argwhere(contact), spacing))
    elif atrium.any():
        rule = "nearest"
        gap = distance_transform_edt(~atrium, sampling=spacing)
        near = edge & (gap <= gap[edge].min() + NEAREST_TOL_MM)
        base = _hinge_midpoint(_mm(np.argwhere(near), spacing))
    else:
        rule = "pca"
        # both ends sit on the principal axis through the centroid; taking
        # the extreme pixels themselves would tilt the axis towards a corner
        centre = pixels.mean(axis=0)
        direction = np.linalg.svd(pixels - centre, full_matrices=False)[2][0]
        proj = (pixels - centre) @ direction
        base = centre + direction * proj.min()
        apex = centre + direction * proj.max()
        return {"apex": apex, "base": base, "length_mm": float(np.linalg.norm(apex - base)),
                "rule": rule}
    apex = pixels[np.argmax(np.linalg.norm(pixels - base, axis=1))]
    return {"apex": apex, "base": base, "length_mm": float(np.linalg.norm(apex - base)),
            "rule": rule}


def disc_diameters(mask: np.ndarray, axis: dict, spacing, n: int = N_DISCS) -> np.ndarray:
    # chord of the mask perpendicular to the long axis at the centre of each
    # of n equal discs from base to apex. the chord is read from a strip two
    # pixels thick so boundary quantisation does not leave a disc empty; one
    # pixel footprint is added so a k-pixel chord measures k pixels
    pixels = _mm(np.argwhere(np.asarray(mask, dtype=bool)), spacing)
    base, apex = axis["base"], axis["apex"]
    length = axis["length_mm"]
    direction = (apex - base) / length
    perp = np.array([-direction[1], direction[0]])
    rel = pixels - base
    t = rel @ direction
    u = rel @ perp
    px = float(np.mean(spacing))
    h = length / n
    diameters = np.zeros(n)
    for i in range(n):
        strip = np.abs(t - (i + 0.5) * h) <= px
        if strip.any():
            diameters[i] = u[strip].max() - u[strip].min() + px
    return diameters


def single_plane_volume(cavity, atrium, spacing, n: int = N_DISCS) -> dict:
    axis = long_axis(cavity, atrium, spacing)
    d = disc_diameters(cavity, axis, spacing, n)
    volume_mm3 = np.pi / 4 * (axis["length_mm"] / n) * np.sum(d**2)
    return {"volume_ml": float(volume_mm3 / 1000), "length_mm": axis["length_mm"],
            "rule": axis["rule"], "diameters_mm": d}


def biplane_volume(cavity_2ch, atrium_2ch, spacing_2ch, cavity_4ch, atrium_4ch, spacing_4ch,
                   n: int = N_DISCS) -> dict:
    # V = pi/4 * (L/n) * sum(a_i * b_i), with L the mean of the two view
    # lengths and a_i, b_i the disc diameters of each view
    two = single_plane_volume(cavity_2ch, atrium_2ch, spacing_2ch, n)
    four = single_plane_volume(cavity_4ch, atrium_4ch, spacing_4ch, n)
    length = (two["length_mm"] + four["length_mm"]) / 2
    volume_mm3 = np.pi / 4 * (length / n) * np.sum(two["diameters_mm"] * four["diameters_mm"])
    return {
        "volume_ml": float(volume_mm3 / 1000),
        "length_mm": length,
        "rules": {"2CH": two["rule"], "4CH": four["rule"]},
        "single_plane_ml": {"2CH": two["volume_ml"], "4CH": four["volume_ml"]},
    }


def ef_from_masks(labels: dict, spacings: dict, cavity_id: int, atrium_id: int) -> dict:
    # labels and spacings are keyed by (view, phase) with views 2CH and 4CH
    # and phases ED and ES; labels are integer label maps on the native grid
    volumes = {}
    for phase in ("ED", "ES"):
        m2, m4 = labels[("2CH", phase)], labels[("4CH", phase)]
        volumes[phase] = biplane_volume(
            m2 == cavity_id, m2 == atrium_id, spacings[("2CH", phase)],
            m4 == cavity_id, m4 == atrium_id, spacings[("4CH", phase)],
        )
    edv, esv = volumes["ED"]["volume_ml"], volumes["ES"]["volume_ml"]
    rules = {f"{v}_{p}": volumes[p]["rules"][v] for p in ("ED", "ES") for v in ("2CH", "4CH")}
    return {
        "edv_ml": edv,
        "esv_ml": esv,
        "ef": float(100 * (edv - esv) / edv) if edv > 0 else float("nan"),
        "rules": rules,
        "n_fallback": sum(r != "annulus" for r in rules.values()),
        "single_plane_ml": {p: volumes[p]["single_plane_ml"] for p in ("ED", "ES")},
    }
