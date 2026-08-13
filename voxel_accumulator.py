"""Online confidence-weighted voxel fusion.

The batch path concatenates every observation from every frame and runs one
``np.unique`` at the end — 93M rows for a 348-frame clip. That is fine when the
clip is already on disk, but a live session cannot hold every observation in
memory (93M x (3 float32 xyz + 3 uint8 rgb + 1 float32 conf) is ~1.6GB and grows
without bound), and it cannot show anything until the walk is over.

This keeps the SAME maths — position and colour are the confidence-weighted mean
of every view of a voxel — but folds each frame in as it arrives, so memory is
proportional to occupied voxels rather than observations, and each frame can
report which voxels it created for immediate display.

Correctness notes, both learned the hard way in the batch implementation:

* The voxel key must be EXACT. An XOR-of-primes hash is not injective, so
  distinct cells silently fuse as if they were one surface, and the collision
  rate grows as the voxel size shrinks.
* The key origin must be GLOBAL and FIXED. Batch code does ``vox -= vox.min()``,
  which is per-call: reusing that incrementally would re-base the grid on every
  frame and misalign each one against the last.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Signed voxel coordinates are packed into one int64. SPAN**3 must stay inside
# int64, so with SPAN = 2**20 the budget is 2**60 — comfortably clear. The
# representable range is +/-524,288 voxels, i.e. +/-2.6km at a 5mm grid.
_KEY_OFFSET = 1 << 19
_KEY_SPAN = 1 << 20


def _encode(vox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Voxel coords [N,3] int64 → (keys [M], in-range mask [N]).

    Out-of-range cells are dropped rather than wrapped: a single bad depth can
    project a point kilometres away, and silently aliasing it onto an occupied
    cell would corrupt real geometry.
    """
    v = vox + _KEY_OFFSET
    ok = np.all((v >= 0) & (v < _KEY_SPAN), axis=1)
    v = v[ok]
    return (v[:, 0] * _KEY_SPAN + v[:, 1]) * _KEY_SPAN + v[:, 2], ok


class VoxelAccumulator:
    """Running confidence-weighted mean per occupied voxel.

    Storage is split into a large sorted ``main`` array and a small sorted
    ``pending`` one. Lookups binary-search both; inserts land in ``pending`` and
    are merged into ``main`` only once pending grows past a fraction of it.
    Inserting straight into ``main`` every frame would be O(N) per frame — at
    10M voxels that is ~80MB of copying per frame, which alone would cap us
    well under the 17fps the model now sustains. Amortising makes it O(1) per
    element at the cost of one extra binary search.
    """

    def __init__(self, voxel_size: float, merge_ratio: float = 0.1,
                 merge_floor: int = 500_000) -> None:
        self.voxel_size = float(voxel_size)
        self._merge_ratio = merge_ratio
        self._merge_floor = merge_floor
        self.dropped = 0          # observations outside the representable grid
        self.observations = 0     # total folded in, for parity with batch logs
        self._main = _Store()
        self._pending = _Store()

    # ── ingestion ────────────────────────────────────────────────────────────
    def add(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        conf: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fold one frame in. Returns (xyz, rgb, conf) of the voxels it CREATED.

        Only created voxels come back because the viewer appends rather than
        indexes — re-sending a refined voxel would draw it twice. Their values
        are this frame's own weighted mean, which is the best estimate that
        exists at the moment of creation; later views refine it and ``finalize``
        reflects that, which is what the finished map is built from.
        """
        empty = (np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8),
                 np.empty(0, np.float32))
        if xyz.shape[0] == 0:
            return empty
        vox = np.floor(np.asarray(xyz, dtype=np.float64) / self.voxel_size).astype(np.int64)
        keys, ok = _encode(vox)
        self.dropped += int((~ok).sum())
        if keys.size == 0:
            return empty
        w = np.asarray(conf, dtype=np.float64)[ok]
        p = np.asarray(xyz, dtype=np.float64)[ok]
        c = np.asarray(rgb, dtype=np.float64)[ok]
        self.observations += int(keys.size)

        # Collapse within the frame first: one binary search per distinct cell
        # instead of one per pixel, which is ~10x fewer for typical depth maps.
        uk, inv = np.unique(keys, return_inverse=True)
        n = uk.size
        fw = np.bincount(inv, weights=w, minlength=n)
        fxyz = np.stack([np.bincount(inv, weights=p[:, d] * w, minlength=n) for d in range(3)], axis=1)
        frgb = np.stack([np.bincount(inv, weights=c[:, d] * w, minlength=n) for d in range(3)], axis=1)
        fcnt = np.bincount(inv, minlength=n).astype(np.int64)

        rest = self._main.accumulate(uk, fw, fxyz, frgb, fcnt)
        if rest is not None:
            uk, fw, fxyz, frgb, fcnt = rest
            rest = self._pending.accumulate(uk, fw, fxyz, frgb, fcnt)
        if rest is None:
            return empty

        uk, fw, fxyz, frgb, fcnt = rest
        self._pending.insert(uk, fw, fxyz, frgb, fcnt)
        if self._pending.size > max(self._merge_floor, int(self._merge_ratio * self._main.size)):
            self._main.absorb(self._pending)
            self._pending = _Store()
        safe = np.where(fw > 0, fw, 1.0)[:, None]
        return ((fxyz / safe).astype(np.float32),
                (frgb / safe).clip(0, 255).astype(np.uint8),
                (fw / np.maximum(fcnt, 1)).astype(np.float32))

    # ── readout ──────────────────────────────────────────────────────────────
    @property
    def size(self) -> int:
        return self._main.size + self._pending.size

    def finalize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All voxels → (xyz float32 [N,3], rgb uint8 [N,3], conf float32 [N]).

        Matches the batch path's outputs exactly: mean position and colour are
        weighted by confidence, and the reported confidence is the MEAN weight
        (sum/count), not the sum — a voxel seen twice at conf 3 scores 3, not 6.
        """
        if self._pending.size:
            self._main.absorb(self._pending)
            self._pending = _Store()
        return self._main.readout()


class _Store:
    """Sorted key array plus parallel weighted sums. Not used directly."""

    __slots__ = ("keys", "sw", "sxyz", "srgb", "cnt")

    def __init__(self) -> None:
        self.keys = np.empty(0, dtype=np.int64)
        self.sw = np.empty(0, dtype=np.float64)
        self.sxyz = np.empty((0, 3), dtype=np.float64)
        self.srgb = np.empty((0, 3), dtype=np.float64)
        self.cnt = np.empty(0, dtype=np.int64)

    @property
    def size(self) -> int:
        return int(self.keys.size)

    def accumulate(self, uk, fw, fxyz, frgb, fcnt):
        """Add to already-known cells. Returns the leftovers, or None if empty."""
        if self.keys.size == 0:
            return (uk, fw, fxyz, frgb, fcnt)
        idx = np.searchsorted(self.keys, uk)
        idx_clipped = np.minimum(idx, self.keys.size - 1)
        hit = self.keys[idx_clipped] == uk
        if hit.any():
            at = idx_clipped[hit]
            # Plain `+=`, not np.add.at: `uk` comes from np.unique and our keys
            # are unique, so `at` can never repeat and the unbuffered form is
            # safe — and about 10x faster, which matters at 100k cells a frame.
            self.sw[at] += fw[hit]
            self.sxyz[at] += fxyz[hit]
            self.srgb[at] += frgb[hit]
            self.cnt[at] += fcnt[hit]
        miss = ~hit
        if not miss.any():
            return None
        return (uk[miss], fw[miss], fxyz[miss], frgb[miss], fcnt[miss])

    def insert(self, uk, fw, fxyz, frgb, fcnt) -> None:
        """Insert brand-new cells, preserving sort order."""
        pos = np.searchsorted(self.keys, uk)
        self.keys = np.insert(self.keys, pos, uk)
        self.sw = np.insert(self.sw, pos, fw)
        self.sxyz = np.insert(self.sxyz, pos, fxyz, axis=0)
        self.srgb = np.insert(self.srgb, pos, frgb, axis=0)
        self.cnt = np.insert(self.cnt, pos, fcnt)

    def absorb(self, other: "_Store") -> None:
        """Fold another store in — its keys are disjoint from ours by
        construction, since anything present here was accumulated, not queued."""
        if other.size == 0:
            return
        self.insert(other.keys, other.sw, other.sxyz, other.srgb, other.cnt)

    def readout(self):
        if self.size == 0:
            return (np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8),
                    np.empty(0, np.float32))
        safe = np.where(self.sw > 0, self.sw, 1.0)[:, None]
        xyz = (self.sxyz / safe).astype(np.float32)
        rgb = (self.srgb / safe).clip(0, 255).astype(np.uint8)
        conf = (self.sw / np.maximum(self.cnt, 1)).astype(np.float32)
        return xyz, rgb, conf


def fuse_batch(
    xyz: np.ndarray, rgb: np.ndarray, conf: np.ndarray, voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-shot fusion over pre-concatenated observations.

    Kept so the harness can compare the incremental path against a single
    reference implementation rather than against engine.py's inlined copy.
    """
    acc = VoxelAccumulator(voxel_size)
    acc.add(xyz, rgb, conf)
    return acc.finalize()
