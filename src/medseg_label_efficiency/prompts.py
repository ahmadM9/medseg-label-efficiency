import zlib

import numpy as np
from scipy.ndimage import distance_transform_edt

# prompts for the promptable arms are derived from the ground-truth mask of
# one structure, an oracle upper bound stated as such wherever reported


def box_from_mask(
    mask: np.ndarray, jitter_px: int = 5, rng: np.random.Generator | None = None
) -> tuple[int, int, int, int] | None:
    # tight box with each edge shifted by a few pixels, like an imprecise
    # human drawing; (row0, col0, row1, col1), None for an empty mask
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
    # the pixel deepest inside the mask, never the center of mass: for a
    # ring like the myocardium the center of mass lies in the cavity
    if not mask.any():
        return None
    depth = distance_transform_edt(mask)
    return tuple(int(v) for v in np.unravel_index(np.argmax(depth), mask.shape))


def sample_rng(*parts) -> np.random.Generator:
    # seeded with crc32, not python's hash(): string hashing is randomized
    # per process, which silently changed prompt jitter between runs
    seed = zlib.crc32("|".join(str(p) for p in parts).encode())
    return np.random.default_rng(seed)
