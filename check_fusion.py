"""Prove the incremental accumulator matches batch fusion, and that it is fast
enough to run inside the live loop.

Run: python check_fusion.py

There is no GPU or checkpoint involved — this exercises fusion alone, which is
the part being rewritten. The world-transform bug survived for days because
nothing compared the new path against the old one numerically; this is that
comparison, written before the live wiring rather than after it.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from voxel_accumulator import VoxelAccumulator

VOXEL = 0.005


def batch_reference(xyz, rgb, conf, voxel_size=VOXEL):
    """engine.py's fusion, transcribed verbatim.

    Deliberately a copy rather than an import: the point is to detect the
    incremental path drifting from the algorithm that produced the maps we have
    already validated by eye, so this must not change when engine.py does.
    """
    vox = np.floor(xyz / voxel_size).astype(np.int64)
    vox = vox - vox.min(axis=0)
    ext = vox.max(axis=0) + 1
    keys = (vox[:, 0] * ext[1] + vox[:, 1]) * ext[2] + vox[:, 2]
    _, inv = np.unique(keys, return_inverse=True)
    inv = inv.reshape(-1)
    n = inv.max() + 1 if inv.size else 0
    w = conf.astype(np.float64)
    wsum = np.bincount(inv, weights=w, minlength=n)
    safe = np.where(wsum > 0, wsum, 1.0)
    fx = np.stack([np.bincount(inv, weights=xyz[:, d].astype(np.float64) * w,
                               minlength=n) / safe for d in range(3)], axis=1)
    fc = np.stack([np.bincount(inv, weights=rgb[:, d].astype(np.float64) * w,
                               minlength=n) / safe for d in range(3)], axis=1)
    counts = np.bincount(inv, minlength=n)
    return (fx.astype(np.float32),
            fc.clip(0, 255).astype(np.uint8),
            (wsum / np.maximum(counts, 1)).astype(np.float32))


def make_frames(rng, n_frames=40, per_frame=20_000, views=10):
    """Synthetic walk over a fixed surface, at REAL observation density.

    ``views`` is observations per voxel, and it is the parameter that decides
    whether this benchmark means anything: measured runs fuse ~93M observations
    into ~9.6M voxels, i.e. 9.8 views each. Generating points uniformly at
    random instead gives ~1.9 views each, which quintuples the number of
    brand-new cells per frame and makes insert cost look far worse than it is.

    The walk is modelled as a window sliding along a pool of surface points:
    each frame samples from the window, so recent surface is re-observed and new
    surface appears gradually, exactly as a corridor does.
    """
    advance = max(1, per_frame // views)
    window = min(per_frame * 3, per_frame * views)
    pool_n = advance * max(0, n_frames - 1) + window
    pool = rng.uniform(-3.0, 3.0, size=(pool_n, 3))
    frames = []
    for i in range(n_frames):
        lo = i * advance
        idx = rng.integers(lo, min(lo + window, pool_n), size=per_frame)
        # Jitter re-observations by well under a voxel: same cell, different
        # float values, so a weighting error cannot hide behind identical input.
        pts = pool[idx] + rng.normal(0, VOXEL * 0.2, (per_frame, 3))
        rgb = rng.integers(0, 256, size=(per_frame, 3)).astype(np.float64)
        conf = rng.uniform(1.0, 25.0, size=per_frame)
        frames.append((pts, rgb, conf))
    return frames


def main() -> int:
    # Defaults are small so the correctness check stays quick. Pass the real
    # shape to measure the live budget:
    #   python check_fusion.py 348 152000
    # (348 frames x 518x294 pixels is what a 35s landscape walk produces.)
    n_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    per_frame = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
    rng = np.random.default_rng(7)
    frames = make_frames(rng, n_frames=n_frames, per_frame=per_frame)

    all_xyz = np.concatenate([f[0] for f in frames])
    all_rgb = np.concatenate([f[1] for f in frames])
    all_conf = np.concatenate([f[2] for f in frames])

    t0 = time.perf_counter()
    bx, bc, bf = batch_reference(all_xyz, all_rgb, all_conf)
    t_batch = time.perf_counter() - t0

    # Fusing every single frame pays np.unique and a sorted insert per frame.
    # Grouping a few frames per call amortises both and recovers most of batch
    # throughput, at the cost of emission latency — at 17fps a group of 8 is
    # 0.47s, which still reads as live to a person walking.
    group = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    acc = VoxelAccumulator(VOXEL)
    created = 0
    t0 = time.perf_counter()
    per_call_ms = []
    for i in range(0, len(frames), group):
        batch = frames[i:i + group]
        gx = np.concatenate([f[0] for f in batch])
        gr = np.concatenate([f[1] for f in batch])
        gc = np.concatenate([f[2] for f in batch])
        t1 = time.perf_counter()
        created += acc.add(gx, gr, gc).size
        per_call_ms.append((time.perf_counter() - t1) * 1e3)
    ix, ic, if_ = acc.finalize()
    t_inc = time.perf_counter() - t0
    per_frame_ms = [m / group for m in per_call_ms]

    print(f"observations {all_xyz.shape[0]:,} | batch {bx.shape[0]:,} voxels "
          f"in {t_batch * 1e3:.0f}ms | incremental {ix.shape[0]:,} voxels "
          f"in {t_inc * 1e3:.0f}ms")
    print(f"group={group} | per-frame amortised: mean {np.mean(per_frame_ms):.1f}ms  "
          f"max {np.max(per_frame_ms):.1f}ms  (budget at 17fps = 59ms) | "
          f"per-call max {np.max(per_call_ms):.0f}ms")
    print(f"voxels created and emitted live: {created:,} "
          f"(should equal final count: {ix.shape[0]:,})")

    ok = True

    # Both key schemes are lexicographic in (x, y, z), so both outputs come out
    # in the same order and can be compared row-wise without re-sorting.
    if bx.shape != ix.shape:
        print(f"FAIL count: batch {bx.shape} vs incremental {ix.shape}")
        return 1

    dp = np.abs(bx - ix).max()
    dc = np.abs(bc.astype(int) - ic.astype(int)).max()
    df = np.abs(bf - if_).max()
    # Tolerances cover float64 summation order only: batch sums a whole column
    # at once, incremental sums frame by frame.
    print(f"max delta — position {dp:.3e}m  colour {dc}  confidence {df:.3e}")
    if dp > 1e-5:
        print("FAIL position drift")
        ok = False
    if dc > 1:
        print("FAIL colour drift")
        ok = False
    if df > 1e-3:
        print("FAIL confidence drift")
        ok = False
    if created != ix.shape[0]:
        print("FAIL emitted-voxel count does not match the final cloud")
        ok = False

    # A voxel far outside the representable grid must be dropped, not wrapped
    # onto an occupied cell.
    far = VoxelAccumulator(VOXEL)
    far.add(np.array([[0.0, 0.0, 0.0], [1e9, 0.0, 0.0]]),
            np.zeros((2, 3)), np.ones(2))
    if far.size != 1 or far.dropped != 1:
        print(f"FAIL out-of-range handling: size {far.size} dropped {far.dropped}")
        ok = False

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
