"""Prompt simulation for the zero-shot promptable-model arms.

Prompts are derived from the ground-truth mask of one structure — an oracle
upper bound, stated as such wherever results are reported. Two styles:

- jittered bounding box: the tight box around the structure with each edge
  shifted by a few pixels, mimicking an imprecise human drawing
- interior point: the pixel deepest inside the structure (maximum of the
  distance transform). Never the center of mass: for ring-shaped structures
  like the myocardium the center of mass lies in the enclosed cavity, which
  is a different structure entirely.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def box_from_mask(
    mask: np.ndarray, jitter_px: int = 5, rng: np.random.Generator | None = None
) -> tuple[int, int, int, int] | None:
    """Jittered tight box around a binary mask, as (row0, col0, row1, col1).

    Returns None for an empty mask. Bounds are clipped to the image, and the
    box always still contains the tight box's center.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    if rng is not None and jitter_px > 0:
        j = rng.integers(-jitter_px, jitter_px + 1, size=4)
        r0, c0, r1, c1 = r0 + j[0], c0 + j[1], r1 + j[2], c1 + j[3]
    h, w = mask.shape
    r0, r1 = np.clip([r0, r1], 0, h - 1)
    c0, c1 = np.clip([c0, c1], 0, w - 1)
    return int(min(r0, r1)), int(min(c0, c1)), int(max(r0, r1)), int(max(c0, c1))


def point_from_mask(mask: np.ndarray) -> tuple[int, int] | None:
    """The (row, col) of the pixel deepest inside the mask; None if empty."""
    if not mask.any():
        return None
    depth = distance_transform_edt(mask)
    return tuple(int(v) for v in np.unravel_index(np.argmax(depth), mask.shape))


def sample_rng(patient: str, structure: int) -> np.random.Generator:
    """Reproducible per-(patient, structure) randomness for box jitter."""
    seed = abs(hash((patient, structure))) % (2**32)
    return np.random.default_rng(seed)
