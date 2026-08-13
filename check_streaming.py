"""Prove the streaming path reconstructs identically to the batch path.

Run on the pod (needs the checkpoint and a GPU):

    python check_streaming.py /path/to/clip.mp4

Batch calls inference_streaming once over the whole clip. Streaming drives
model.forward() frame by frame from outside. They should agree exactly — the
per-frame output of inference_streaming is final and never revised, so driving
the same calls in the same order must give the same numbers.

If they disagree, the streaming path is wrong and nothing downstream of it is
trustworthy. This is the check that did not exist when the inverted world
transform shipped.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

import engine
from streaming import StreamingReconstructor


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]

    # The KV pool alone is ~6.5GiB at landscape and ~11.8GiB at portrait, so a
    # server left running by nohup (Ctrl-C only kills the tail, not the service)
    # leaves nowhere near enough for a second model. Fail here with something
    # readable rather than deep inside a layer norm.
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        if free < 14 * (1 << 30):
            print(f"only {free / 2**30:.1f} GiB of {total / 2**30:.1f} GiB free — "
                  f"something else is holding the GPU.\n"
                  f"  pkill -9 -f uvicorn; sleep 3; "
                  f"nvidia-smi --query-compute-apps=pid,used_memory --format=csv")
            return 2

    images = engine.load_video_frames(path)               # [1,S,3,H,W]
    S = images.shape[1]
    kf = engine.auto_keyframe_interval(S)
    print(f"clip: {S} frames {images.shape[-1]}x{images.shape[-2]} keyframe={kf}")

    model = engine.load_model()

    # ── batch ────────────────────────────────────────────────────────────────
    engine.warm_model(model, images, kf, quiet=True)
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=engine._DTYPE):
        preds = model.inference_streaming(
            images, num_scale_frames=engine.NUM_SCALE_FRAMES,
            keyframe_interval=kf, output_device=torch.device("cpu"))
    t_batch = time.time() - t0
    b_pose = preds["pose_enc"].float()
    b_depth = preds["depth"].float().numpy()[0].squeeze(-1)
    b_conf = preds["depth_conf"].float().numpy()[0]
    engine.release_gpu_memory()

    # ── streaming ────────────────────────────────────────────────────────────
    sr = StreamingReconstructor(
        model, engine._DEVICE, engine._DTYPE,
        scale_frames=engine.NUM_SCALE_FRAMES, keyframe_interval=kf,
        conf_threshold=engine.CONF_THRESHOLD)
    t0 = time.time()
    emitted = 0
    first_at = None
    for i in range(S):
        for r in sr.push(images[0, i]):
            emitted += r.xyz.shape[0]
            if first_at is None and r.xyz.shape[0]:
                first_at = time.time() - t0
    t_stream = time.time() - t0
    _, s_depth, s_conf, s_pose = sr.retained()

    print(f"batch {t_batch:.1f}s | streaming {t_stream:.1f}s "
          f"(infer {sr.infer_s:.1f}s + fuse {sr.fuse_s:.1f}s)")
    print(f"first points visible after {first_at:.1f}s "
          f"(batch shows nothing until {t_batch:.1f}s)")
    print(f"live preview: {sr.preview_points:,} voxels, {emitted:,} emitted")

    ok = True
    dp = np.abs(b_pose.numpy() - s_pose.numpy()).max()
    dd = np.abs(b_depth - s_depth).max()
    dc = np.abs(b_conf - s_conf).max()
    print(f"max delta — pose {dp:.3e}  depth {dd:.3e}  conf {dc:.3e}")
    # bf16 accumulation is not bit-reproducible across differing block shapes,
    # so allow a small tolerance; anything structural is orders larger.
    if dp > 1e-4:
        print("FAIL pose drift — streaming is NOT equivalent to batch")
        ok = False
    if dd > 1e-2:
        print("FAIL depth drift")
        ok = False
    if dc > 1e-2:
        print("FAIL confidence drift")
        ok = False

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
