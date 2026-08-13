"""Frame-by-frame reconstruction for the live path.

The batch path buffers a whole clip, calls ``inference_streaming`` once, and
emits everything at the end — so the map sits empty for the length of the walk
plus the length of inference. ``inference_streaming`` is itself just a loop over
``model.forward()`` (gct_stream.py:446) whose per-frame output is final and
never revised, so the same work can be driven from outside and each frame's
geometry emitted the moment it exists.

TWO-TIER OUTPUT, and the reason for it:

* **Live preview** — coarse grid, strided pixels, fused incrementally. Measured:
  full-resolution fusion costs ~156ms/frame against a ~59ms budget at 17fps, and
  profiling split that evenly between structure maintenance and inherent work
  (encode + unique + bincounts), so no data structure fixes it — the input has
  to be smaller. Stride 2 on a 2cm grid runs 32ms mean / 51ms max.
* **Final map** — full stride, full-resolution grid, fused in one pass at stop
  from the retained per-frame depth. Matches what the batch path produces today.

Transport agrees with that split independently: a 9.5M-point cloud is ~114MB and
cannot stream live at any fusion speed, while a 2cm room is ~400k voxels ≈ 5MB.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from voxel_accumulator import VoxelAccumulator


@dataclass
class FrameResult:
    """One frame's contribution, ready to send to the viewer."""
    index: int
    pose_c2w: np.ndarray                  # [4,4] camera-to-world
    xyz: np.ndarray                       # [N,3] float32 — NEW preview voxels only
    rgb: np.ndarray                       # [N,3] uint8
    conf: np.ndarray                      # [N]   float32


class StreamingReconstructor:
    """Drives the model one frame at a time and fuses as it goes.

    Mirrors ``inference_streaming``'s call pattern exactly: one bidirectional
    forward over the scale block, then ``num_frame_per_block=1`` per frame with
    the keyframe skip_append policy. ``warm_model`` already exercises that same
    pattern, so the mechanism is proven — what is new is consuming each frame's
    output immediately rather than concatenating.
    """

    def __init__(
        self,
        model,
        device: torch.device,
        dtype: torch.dtype,
        scale_frames: int = 8,
        keyframe_interval: int = 1,
        conf_threshold: float = 1.0,
        preview_stride: int = 2,
        preview_voxel: float = 0.02,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.scale_frames = scale_frames
        self.keyframe_interval = keyframe_interval
        self.conf_threshold = conf_threshold
        self.preview_stride = preview_stride
        self.preview = VoxelAccumulator(preview_voxel)

        self.index = 0
        self.started = False
        self.infer_s = 0.0
        self.fuse_s = 0.0
        self._hw: Optional[Tuple[int, int]] = None
        self._pending: List[torch.Tensor] = []
        # Retained per-frame outputs for the full-quality pass at stop. Depth and
        # confidence are kept rather than re-run: ~600MB for a 35s landscape walk
        # (348 frames x 518x294 x 8 bytes), the same order as the frame buffer the
        # batch path already holds, and far cheaper than a second inference pass.
        self._frames: List[torch.Tensor] = []
        self._depth: List[np.ndarray] = []
        self._conf: List[np.ndarray] = []
        self._pose_enc: List[torch.Tensor] = []

    # ── ingestion ────────────────────────────────────────────────────────────
    def push(self, frame: torch.Tensor) -> List[FrameResult]:
        """Feed one preprocessed frame [3,H,W]. Returns whatever is now ready.

        Nothing comes back until the scale block is full: those first frames
        establish world scale bidirectionally, and until they have run there is
        no pose to unproject anything against.
        """
        if not self.started:
            self._pending.append(frame)
            if len(self._pending) < self.scale_frames:
                return []
            block = torch.stack(self._pending, dim=0).unsqueeze(0)  # [1,S,3,H,W]
            batch_frames = self._pending
            self._pending = []
            self.started = True
            self._hw = tuple(block.shape[-2:])
            self.model.clean_kv_cache()
            return self._consume(self._forward(block, n=self.scale_frames), batch_frames)

        block = frame.unsqueeze(0).unsqueeze(0)  # [1,1,3,H,W]
        is_keyframe = (self.keyframe_interval <= 1) or (
            (self.index - self.scale_frames) % self.keyframe_interval == 0)
        if not is_keyframe:
            self.model._set_skip_append(True)
        try:
            out = self._forward(block, n=1)
        finally:
            # Restored in a finally: leaving skip_append set after an exception
            # would silently stop every later frame entering the KV cache, and
            # the only symptom would be a gradually worse map.
            if not is_keyframe:
                self.model._set_skip_append(False)
        return self._consume(out, [frame])

    def _forward(self, block: torch.Tensor, n: int) -> dict:
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.dtype):
            out = self.model.forward(
                block.to(self.device, non_blocking=True),
                num_frame_for_scale=self.scale_frames,
                num_frame_per_block=n,
                causal_inference=True,
            )
        self.infer_s += time.time() - t0
        return out

    def _consume(self, out: dict, frames: List[torch.Tensor]) -> List[FrameResult]:
        """Unproject each frame in the block, fuse into the preview, emit."""
        # Imported here, not at module scope: engine imports this module for the
        # live path, so a top-level import would be circular.
        from engine import poses_from_pose_enc
        from lingbot_map.utils.geometry import depth_to_world_coords_points

        pose_enc = out["pose_enc"].detach().float().cpu()      # [1,n,9]
        depth = out["depth"].detach().float().cpu()            # [1,n,H,W,1]
        conf = out["depth_conf"].detach().float().cpu()        # [1,n,H,W]
        w2c, intri, c2w = poses_from_pose_enc(pose_enc, self._hw)

        t0 = time.time()
        results: List[FrameResult] = []
        st = self.preview_stride
        for i, frame in enumerate(frames):
            d = depth[0, i].numpy().squeeze(-1)                # [H,W]
            c = conf[0, i].numpy()                             # [H,W]
            rgb = (frame.float().numpy().transpose(1, 2, 0) * 255
                   ).clip(0, 255).astype(np.uint8)             # [H,W,3]

            self._frames.append(frame)
            self._depth.append(d)
            self._conf.append(c)
            self._pose_enc.append(pose_enc[:, i:i + 1])

            wp, _, _ = depth_to_world_coords_points(d, w2c[i], intri[i])
            wp_s = wp[::st, ::st].reshape(-1, 3)
            c_s = c[::st, ::st].reshape(-1)
            rgb_s = rgb[::st, ::st].reshape(-1, 3).astype(np.float64)
            keep = (c_s >= self.conf_threshold) & np.isfinite(wp_s).all(axis=1)

            idx = self.index
            self.index += 1
            if not keep.any():
                results.append(FrameResult(
                    idx, c2w[i], np.empty((0, 3), np.float32),
                    np.empty((0, 3), np.uint8), np.empty(0, np.float32)))
                continue
            nx, nr, nc = self.preview.add(wp_s[keep], rgb_s[keep], c_s[keep])
            results.append(FrameResult(idx, c2w[i], nx, nr, nc))
        self.fuse_s += time.time() - t0
        return results

    # ── readout ──────────────────────────────────────────────────────────────
    @property
    def preview_points(self) -> int:
        return self.preview.size

    def retained(self) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, torch.Tensor]:
        """Everything the full-quality pass needs: frames, depth, conf, poses.

        Returned rather than fused here so the caller can hand it to the same
        fusion code the batch path uses, instead of this module growing a second
        implementation that can drift from it.
        """
        images = torch.stack(self._frames, dim=0).unsqueeze(0)   # [1,S,3,H,W]
        pose_enc = torch.cat(self._pose_enc, dim=1)              # [1,S,9]
        return (images,
                np.stack(self._depth, axis=0),
                np.stack(self._conf, axis=0),
                pose_enc)
