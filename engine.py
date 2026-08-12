"""Live mapping engine — wraps lingbot-map's streaming loop for frame-by-frame
HTTP feeding (the "map builder" behind shipment-glasses' Build-3D-Map button).

Mirrors the exact call pattern of GCTStream.inference_streaming / demo.py's
_warm_streaming, but driven by frames arriving over the network instead of a
preloaded tensor:

    session start → buffer first NUM_SCALE_FRAMES frames ("warming")
    Phase 1       → one forward over the scale block (establishes world scale)
    per frame     → forward(num_frame_per_block=1), keyframe KV policy
    per frame out → world_points (already global coords!) + conf filter
                    → pixel stride + voxel dedupe → chunk (only NEW voxels)
    stop          → PLY dump of the accumulated cloud

Design notes:
- Model loaded ONCE at import/boot; ONE active session at a time (Phase A).
- The model is not thread-safe: all inference runs under a lock.
- bf16 aggregator cast + expandable_segments (set in server.py before torch
  import) keep 4GB laptop GPUs viable at reduced resolution.
"""

from __future__ import annotations

import io
import os
import struct
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.geometry import (
    closed_form_inverse_se3_general,
    depth_to_world_coords_points,
)
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

# ── Config (env) ─────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "lingbot-map.pt"))
# MUST stay 518 — the checkpoint/architecture is fixed-size (no pos-embed
# interpolation; smaller inputs break internal token reshapes). VRAM control
# comes from bf16 + kv sliding window + camera_iters=1 instead.
IMG_SIZE = int(os.environ.get("MAP_IMG_SIZE", "518"))
NUM_SCALE_FRAMES = int(os.environ.get("MAP_SCALE_FRAMES", "8"))
# The model's RoPE is trained to ~320 frames. Past that the KV cache runs
# outside the trained positional range and pose COLLAPSES (drifting depth,
# confidence pinned at its floor). demo.py guards against this by auto-picking
# ceil(N/320) so the cache stays inside the limit while every frame is still
# inferred. 0 = auto (default); set a positive value to force one.
KEYFRAME_INTERVAL_ENV = int(os.environ.get("MAP_KEYFRAME_INTERVAL", "0"))
ROPE_FRAME_LIMIT = 320


def auto_keyframe_interval(num_frames: int) -> int:
    """demo.py's rule: 1 up to the RoPE limit, else ceil(N / 320)."""
    if KEYFRAME_INTERVAL_ENV > 0:
        return KEYFRAME_INTERVAL_ENV
    if num_frames > ROPE_FRAME_LIMIT:
        return (num_frames + ROPE_FRAME_LIMIT - 1) // ROPE_FRAME_LIMIT
    return 1
CAMERA_ITERS = int(os.environ.get("MAP_CAMERA_ITERS", "4"))  # demo default; 1 = fast/inaccurate
CONF_THRESHOLD = float(os.environ.get("MAP_CONF_THRESHOLD", "1.5"))
PIXEL_STRIDE = int(os.environ.get("MAP_PIXEL_STRIDE", "4"))    # sample every Nth pixel
# Fusion grid. Measured on a 348-frame clip (93.4M observations): 0.03 gave
# 168k points at 555 views each (heavily over-smoothed), 0.012 gave 1.6M at 57,
# and 0.005 gives 9.6M at ~10 views — enough averaging to cancel noise without
# erasing detail. Go smaller for more detail at the cost of noise and memory.
VOXEL_SIZE = float(os.environ.get("MAP_VOXEL_SIZE", "0.005"))
MAX_FRAMES = int(os.environ.get("MAP_MAX_FRAMES", "2000"))
MAPS_DIR = os.environ.get("MAPS_DIR", os.path.join(os.path.dirname(__file__), "maps"))
# auto = crop for landscape, pad for portrait — the only correct default.
# crop resizes width to IMG_SIZE and centre-crops the overflow, so it discards
# nothing for a LANDSCAPE frame (518x294 is close to the model's native
# 518x378) but throws away ~44% of a PORTRAIT frame's vertical FOV, including
# the near-floor region that carries the most parallax. Measured on identical
# portrait frames: pad gave camera path 6.33 vs 4.75, extent 1.57 vs 0.96 and
# 2.5-4.5x more per-frame depth relief. Padding a landscape frame instead just
# wastes 44% of the tokens on blank borders, hence per-aspect selection.
PREPROCESS_MODE = os.environ.get("MAP_PREPROCESS_MODE", "auto").lower()


def resolve_mode(width: int, height: int) -> str:
    """'auto' → crop for landscape, pad for portrait."""
    if PREPROCESS_MODE in ("crop", "pad"):
        return PREPROCESS_MODE
    return "pad" if height > width else "crop"
# Debug: save every Nth RAW incoming frame (exact pixels the model receives —
# resolution + blur check). Off unless MAP_DEBUG_DUMP is set to a truthy value.
DEBUG_DUMP = os.environ.get("MAP_DEBUG_DUMP", "").lower() not in ("", "0", "false", "no")
DEBUG_EVERY = int(os.environ.get("MAP_DEBUG_EVERY", "10"))
DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug_frames")

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else (torch.float16 if torch.cuda.is_available() else torch.float32)
)

_model: Optional[GCTStream] = None
_infer_lock = threading.Lock()


def load_model() -> GCTStream:
    """Load the checkpoint once (call at server boot)."""
    global _model
    if _model is not None:
        return _model
    t0 = time.time()
    # NOTE: the checkpoint's pos_embed is sized for img_size=518 — the model is
    # ALWAYS built at 518. Reduced VRAM comes from feeding smaller inputs
    # (MAP_IMG_SIZE) — the ViT interpolates position encodings at runtime.
    model = GCTStream(
        img_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=NUM_SCALE_FRAMES,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,               # no FlashInfer/Triton dependency
        camera_num_iterations=CAMERA_ITERS,  # 4 = demo pose accuracy (fuses geometry)
    )
    ckpt = torch.load(MODEL_PATH, map_location=_DEVICE, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(_DEVICE).eval()
    # Same trick as demo.py: bf16 trunk saves ~2-3GB; heads stay fp32 internally.
    if _DTYPE != torch.float32 and getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=_DTYPE)
    _model = model
    print(f"[engine] model loaded in {time.time() - t0:.1f}s "
          f"(device={_DEVICE}, dtype={_DTYPE}, img_size={IMG_SIZE})")
    return model


def _preprocess_jpeg(jpeg_bytes: bytes) -> torch.Tensor:
    """JPEG bytes → model tensor (HTTP debug path). Delegates to the array path
    so every entry point shares one preprocessing rule."""
    img = Image.open(io.BytesIO(jpeg_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    return _preprocess_array(np.asarray(img))


def _preprocess_array(rgb: np.ndarray) -> torch.Tensor:
    """Decoded RGB frame [H,W,3] → model tensor, mirroring
    load_and_preprocess_images without JPEG/disk.

    MAP_PREPROCESS_MODE:
      crop (demo default) — width→IMG_SIZE, centre-crop the overflow. Lossless
        for LANDSCAPE input (nothing to crop), but a PORTRAIT phone stream loses
        ~44% of its vertical field of view, including the near-floor region that
        carries the strongest parallax for pose estimation.
      pad — largest side→IMG_SIZE, letterbox the rest. Keeps the whole frame at
        the cost of blank borders.
    """
    img = Image.fromarray(rgb)
    w, h = img.size
    if resolve_mode(w, h) == "pad":
        if w >= h:
            new_w = IMG_SIZE
            new_h = round(h * (new_w / w) / 14) * 14
        else:
            new_h = IMG_SIZE
            new_w = round(w * (new_h / h) / 14) * 14
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        t = to_tensor(img)
        pad_h = IMG_SIZE - t.shape[1]
        pad_w = IMG_SIZE - t.shape[2]
        if pad_h > 0 or pad_w > 0:
            top = pad_h // 2
            left = pad_w // 2
            t = torch.nn.functional.pad(
                t, (left, pad_w - left, top, pad_h - top), value=1.0)
        return t

    new_w = IMG_SIZE
    new_h = round(h * (new_w / w) / 14) * 14
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    t = to_tensor(img)
    if new_h > IMG_SIZE:
        start_y = (new_h - IMG_SIZE) // 2
        t = t[:, start_y:start_y + IMG_SIZE, :]
    return t


@dataclass
class Chunk:
    seq: int
    count: int
    points: bytes  # Float32 xyz interleaved
    colors: bytes  # Uint8 rgb interleaved
    confs: bytes   # Float32 per-point confidence (viewer visibility threshold)
    pose: list     # 16 floats, c2w row-major (camera trail)


@dataclass
class MapSession:
    session_id: str
    peer_id: str
    state: str = "capturing"   # capturing | processing | stopped
    frames_in: int = 0
    frames_mapped: int = 0
    total_points: int = 0
    seq: int = 0
    # Buffer of preprocessed frames (CPU tensors), assembled into the clip that
    # inference_streaming consumes on stop — same as demo.py loading a video.
    frames_buf: list = field(default_factory=list)
    voxels: set = field(default_factory=set)
    acc_xyz: list = field(default_factory=list)
    acc_rgb: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_frame_at: float = field(default_factory=time.time)

    # ── frame ingestion (buffer only — no inference during capture) ───────
    def add_frame(self, jpeg_bytes: bytes) -> list["Chunk"]:
        """Feed one JPEG (HTTP debug path). Buffers; returns no chunks."""
        if self.state != "capturing" or self.frames_in >= MAX_FRAMES:
            return []
        self.last_frame_at = time.time()
        self.frames_in += 1
        self.frames_buf.append(_preprocess_jpeg(jpeg_bytes))
        return []

    def add_frame_array(self, rgb: np.ndarray) -> list["Chunk"]:
        """Feed one decoded RGB frame (WebRTC path). Buffers; no chunks yet —
        the whole clip is reconstructed together on stop (demo.py parity)."""
        if self.state != "capturing" or self.frames_in >= MAX_FRAMES:
            return []
        self.last_frame_at = time.time()
        self.frames_in += 1
        if DEBUG_DUMP and self.frames_in % DEBUG_EVERY == 0:
            self._dump_frame(rgb)
        self.frames_buf.append(_preprocess_array(rgb))
        return []

    def _dump_frame(self, rgb: np.ndarray) -> None:
        """Save the raw received frame as JPEG — what WebRTC actually delivered
        (resolution in the filename), before any resize the model applies."""
        try:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            h, w = rgb.shape[:2]
            name = f"{self.session_id}_{self.frames_in:05d}_{w}x{h}.jpg"
            Image.fromarray(rgb).save(os.path.join(DEBUG_DIR, name), quality=90)
        except Exception as err:
            print(f"[debug] frame dump failed: {err}")

    # ── reconstruction (demo.py's inference_streaming, run on the clip) ────
    def process(self, images: Optional[torch.Tensor] = None) -> list["Chunk"]:
        """Run the OFFICIAL streaming reconstruction over a clip, exactly like
        demo.py: Phase-1 scale block, then frame-by-frame with the KV cache and
        full camera-pose refinement. Returns per-frame chunks for the viewer.

        `images` [1,S,3,H,W] comes from a decoded video (upload path); when
        omitted the buffered WebRTC frames are used instead."""
        if images is None:
            if not self.frames_buf:
                return []
            # Pad any odd-aspect frame to the modal shape so torch.stack
            # succeeds (phone is portrait → nearly all frames are 518x518).
            shapes = {tuple(f.shape) for f in self.frames_buf}
            if len(shapes) > 1:
                h = max(f.shape[1] for f in self.frames_buf)
                w = max(f.shape[2] for f in self.frames_buf)
                self.frames_buf = [
                    torch.nn.functional.pad(
                        f, (0, w - f.shape[2], 0, h - f.shape[1]), value=1.0)
                    for f in self.frames_buf
                ]
            images = torch.stack(self.frames_buf, dim=0).unsqueeze(0)  # [1,S,3,H,W]
            self.frames_buf = []
        self.state = "processing"
        model = load_model()
        S = images.shape[1]
        keyframe_interval = auto_keyframe_interval(S)
        t0 = time.time()
        print(f"[engine] reconstructing {S} frames (inference_streaming, "
              f"scale={NUM_SCALE_FRAMES}, keyframe={keyframe_interval}"
              f"{' [auto: >320-frame RoPE limit]' if keyframe_interval > 1 else ''}, "
              f"cam_iters={CAMERA_ITERS})…")
        with _infer_lock, torch.no_grad(), torch.amp.autocast("cuda", dtype=_DTYPE):
            preds = model.inference_streaming(
                images,
                num_scale_frames=NUM_SCALE_FRAMES,
                keyframe_interval=keyframe_interval,
                output_device=torch.device("cpu"),
            )
        pose_enc = preds["pose_enc"]        # [1,S,9]
        depth = preds["depth"]              # [1,S,H,W,1]
        depth_conf = preds["depth_conf"]    # [1,S,H,W]
        print(f"[engine] inference done in {time.time() - t0:.1f}s — extracting")

        # EXACT demo.py → viser composition. pose_encoding_to_extri_intri returns
        # the camera-TO-world matrix; demo's postprocess inverts it and the
        # geometry helper inverts it back before applying. Do NOT shortcut this
        # with model._unproject_depth_to_world — that applies the INVERSE
        # transform (points scatter, dedupe never hits, trail collapses).
        hw = tuple(images.shape[-2:])
        extri, intri = pose_encoding_to_extri_intri(pose_enc.float(), hw)  # [1,S,3,4]/[1,S,3,3]
        c2w = torch.zeros((*extri.shape[:-2], 4, 4), dtype=extri.dtype)
        c2w[..., :3, :4] = extri
        c2w[..., 3, 3] = 1.0
        w2c = closed_form_inverse_se3_general(c2w)[..., :3, :4]  # what demo stores as "extrinsic"
        w2c_np = w2c[0].cpu().numpy()
        intri_np = intri[0].cpu().numpy()
        c2w_np = c2w[0].cpu().numpy()

        # ── diagnostics: is depth degenerate, or is camera translation lost? ──
        cap_dur = max(1e-6, self.last_frame_at - self.created_at)
        print(f"[diag] captured {self.frames_in} frames over {cap_dur:.1f}s = "
              f"{self.frames_in / cap_dur:.1f} fps effective "
              f"(walking at 5fps puts ~30cm between frames — too far for the "
              f"model to track)")
        cam_pos = c2w_np[:, :3, 3]                                  # [S,3]
        path_len = float(np.linalg.norm(np.diff(cam_pos, axis=0), axis=1).sum())
        span = float(np.linalg.norm(cam_pos.max(0) - cam_pos.min(0)))
        print(f"[diag] camera path length {path_len:.2f} | bbox span {span:.2f} | "
              f"start {cam_pos[0].round(2)} end {cam_pos[-1].round(2)}")
        for s in (0, S // 2, S - 1):
            d = depth[0, s].float().numpy().squeeze(-1)
            c = depth_conf[0, s].float().numpy()
            print(f"[diag] frame {s}: depth min {d.min():.3f} max {d.max():.3f} "
                  f"mean {d.mean():.3f} std {d.std():.3f} | "
                  f"conf mean {c.mean():.2f} p10 {np.percentile(c, 10):.2f} "
                  f"p90 {np.percentile(c, 90):.2f}")

        # ── pass 1: gather every observation + per-frame camera pose ──────────
        # A pose-only chunk (count 0) carries the trajectory so the viewer's
        # trail and frustums still build frame by frame.
        chunks: list[Chunk] = []
        obs_xyz: list[np.ndarray] = []
        obs_rgb: list[np.ndarray] = []
        obs_conf: list[np.ndarray] = []
        st = PIXEL_STRIDE
        for s in range(S):
            # Per-frame unprojection keeps peak memory at one frame.
            wp, _, _ = depth_to_world_coords_points(
                depth[0, s].float().numpy().squeeze(-1),  # [H,W]
                w2c_np[s], intri_np[s],
            )                                                          # [H,W,3]
            conf = depth_conf[0, s].float().numpy()                    # [H,W]
            rgb = (images[0, s].float().numpy().transpose(1, 2, 0) * 255
                   ).clip(0, 255).astype(np.uint8)                     # [H,W,3]

            wp_s = wp[::st, ::st].reshape(-1, 3)
            conf_s = conf[::st, ::st].reshape(-1)
            rgb_s = rgb[::st, ::st].reshape(-1, 3)
            keep = (conf_s >= CONF_THRESHOLD) & np.isfinite(wp_s).all(axis=1)
            if keep.any():
                obs_xyz.append(wp_s[keep].astype(np.float32))
                obs_rgb.append(rgb_s[keep])
                obs_conf.append(conf_s[keep].astype(np.float32))
            self.frames_mapped += 1
            self.seq += 1
            chunks.append(Chunk(
                seq=self.seq, count=0, points=b"", colors=b"", confs=b"",
                pose=np.asarray(c2w_np[s], dtype=np.float32).reshape(-1).tolist(),
            ))

        # ── pass 2: fuse observations per voxel (confidence-weighted) ─────────
        # Previously the FIRST point to land in a voxel won and every later
        # sighting of that surface was discarded — so each surface kept its
        # earliest, most distant, noisiest observation. Averaging all views of a
        # voxel instead cuts positional noise roughly with sqrt(view count).
        if obs_xyz:
            xyz = np.concatenate(obs_xyz, axis=0)
            rgbs = np.concatenate(obs_rgb, axis=0).astype(np.float32)
            confs = np.concatenate(obs_conf, axis=0)
            del obs_xyz, obs_rgb, obs_conf
            vox = np.floor(xyz / VOXEL_SIZE).astype(np.int64)
            # EXACT voxel key. The usual XOR-of-primes hash is not injective:
            # distinct cells collide and get fused as if they were one surface,
            # and shrinking VOXEL_SIZE makes the coordinates larger and the
            # collisions MORE common — which is why 0.012 produced fewer points
            # than 0.03. A linear index over the actual occupied grid cannot
            # collide; fall back to a true row-wise unique if it would overflow.
            vox -= vox.min(axis=0)
            ext = vox.max(axis=0) + 1
            if float(ext[0]) * float(ext[1]) * float(ext[2]) < 9e18:
                keys = (vox[:, 0] * ext[1] + vox[:, 1]) * ext[2] + vox[:, 2]
                _, inv = np.unique(keys, return_inverse=True)
            else:
                _, inv = np.unique(vox, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            n_vox = inv.max() + 1 if inv.size else 0
            w = confs                                    # weight by confidence
            wsum = np.bincount(inv, weights=w, minlength=n_vox)
            wsum_safe = np.where(wsum > 0, wsum, 1.0)
            fused_xyz = np.stack([
                np.bincount(inv, weights=xyz[:, d] * w, minlength=n_vox) / wsum_safe
                for d in range(3)
            ], axis=1).astype(np.float32)
            fused_rgb = np.stack([
                np.bincount(inv, weights=rgbs[:, d] * w, minlength=n_vox) / wsum_safe
                for d in range(3)
            ], axis=1).clip(0, 255).astype(np.uint8)
            counts = np.bincount(inv, minlength=n_vox)
            fused_conf = (wsum / np.maximum(counts, 1)).astype(np.float32)
            print(f"[engine] fused {xyz.shape[0]} observations into "
                  f"{fused_xyz.shape[0]} voxels "
                  f"({xyz.shape[0] / max(1, fused_xyz.shape[0]):.1f} views each)")
            del xyz, rgbs, confs, vox, keys, inv

            PAGE = 200_000
            for i in range(0, fused_xyz.shape[0], PAGE):
                px = fused_xyz[i:i + PAGE]
                pc = fused_rgb[i:i + PAGE]
                pf = fused_conf[i:i + PAGE]
                self.acc_xyz.append(px)
                self.acc_rgb.append(pc)
                self.total_points += int(px.shape[0])
                self.seq += 1
                chunks.append(Chunk(
                    seq=self.seq, count=int(px.shape[0]),
                    points=px.tobytes(), colors=pc.tobytes(),
                    confs=pf.tobytes(), pose=[],
                ))
        if self.acc_xyz:
            allp = np.concatenate(self.acc_xyz, axis=0)
            lo, hi = allp.min(0), allp.max(0)
            print(f"[diag] cloud bbox {(hi - lo).round(2)} "
                  f"(camera path {path_len:.2f} — a real walk should be the "
                  f"same order as the scene size)")
        print(f"[engine] extracted {self.total_points} points in {len(chunks)} chunks")
        return chunks

    def _points_to_chunk(
        self, wp: np.ndarray, conf: np.ndarray, rgb: np.ndarray, c2w: np.ndarray,
    ) -> Optional[Chunk]:
        """One frame's world points → stride → conf filter → voxel dedupe → chunk."""
        self.frames_mapped += 1
        st = PIXEL_STRIDE
        wp_s = wp[::st, ::st].reshape(-1, 3)
        conf_s = conf[::st, ::st].reshape(-1)
        rgb_s = rgb[::st, ::st].reshape(-1, 3)

        keep = conf_s >= CONF_THRESHOLD
        wp_s, rgb_s, conf_s = wp_s[keep], rgb_s[keep], conf_s[keep]
        if wp_s.shape[0] == 0:
            return None

        vox = np.floor(wp_s / VOXEL_SIZE).astype(np.int64)
        keys = vox[:, 0] * 73856093 ^ vox[:, 1] * 19349663 ^ vox[:, 2] * 83492791
        new_mask = np.fromiter(
            (k not in self.voxels for k in keys), dtype=bool, count=len(keys)
        )
        self.voxels.update(keys[new_mask].tolist())
        wp_new, rgb_new, conf_new = wp_s[new_mask], rgb_s[new_mask], conf_s[new_mask]
        if wp_new.shape[0] == 0:
            return None

        # Camera-to-world for the trail — same matrix the viser viewer uses for
        # frustums (cam_to_world_mat = inverse of the stored "extrinsic").
        pose_mat = np.asarray(c2w, dtype=np.float32).reshape(-1).tolist()

        self.acc_xyz.append(wp_new.astype(np.float32))
        self.acc_rgb.append(rgb_new)
        self.total_points += int(wp_new.shape[0])
        self.seq += 1
        return Chunk(
            seq=self.seq,
            count=int(wp_new.shape[0]),
            points=wp_new.astype(np.float32).tobytes(),
            colors=rgb_new.astype(np.uint8).tobytes(),
            confs=conf_new.astype(np.float32).tobytes(),
            pose=pose_mat,
        )

    # ── finalize ─────────────────────────────────────────────────────────
    def finalize(self) -> dict:
        self.state = "stopped"
        model = _model
        if model is not None:
            with _infer_lock:
                model.clean_kv_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ply_path = ""
        if self.acc_xyz:
            os.makedirs(MAPS_DIR, exist_ok=True)
            xyz = np.concatenate(self.acc_xyz, axis=0)
            rgb = np.concatenate(self.acc_rgb, axis=0)
            ply_path = os.path.join(MAPS_DIR, f"{self.session_id}.ply")
            _write_ply(ply_path, xyz, rgb)

        return {
            "sessionId": self.session_id,
            "peerId": self.peer_id,
            "frames": self.frames_mapped,
            "points": self.total_points,
            "durationSec": round(time.time() - self.created_at, 1),
            "ply": ply_path,
        }


def _write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Binary little-endian PLY (opens in MeshLab/CloudCompare/three.js)."""
    n = xyz.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        for i in range(n):
            f.write(struct.pack("<fffBBB", xyz[i, 0], xyz[i, 1], xyz[i, 2],
                                rgb[i, 0], rgb[i, 1], rgb[i, 2]))


def load_video_frames(video_path: str, fps: int = 10) -> torch.Tensor:
    """Decode an uploaded video into demo.py's preprocessed tensor.

    Byte-for-byte demo.py's path: sample every Nth frame to hit `fps`, write
    JPEGs, then run them through load_and_preprocess_images. Uploaded video
    keeps its full resolution — no WebRTC downscaling — which is exactly what
    the reconstruction needs to recover camera motion.
    """
    import cv2  # heavy import, only needed for the upload path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("could not open video")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, round(src_fps / max(1, fps)))
    out_dir = tempfile.mkdtemp(prefix="lingbot_video_")
    paths: list[str] = []
    idx = 0
    src_w = src_h = 0
    try:
        while len(paths) < MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % interval == 0:
                if not paths:
                    src_h, src_w = frame.shape[:2]
                p = os.path.join(out_dir, f"{len(paths):06d}.jpg")
                cv2.imwrite(p, frame)
                paths.append(p)
            idx += 1
    finally:
        cap.release()
    if not paths:
        raise ValueError("no frames decoded from video")
    # Match the live path: crop a landscape recording (lossless), pad a portrait
    # one (crop would discard ~44% of its vertical FOV).
    mode = resolve_mode(src_w, src_h)
    print(f"[engine] decoded {len(paths)} frames from video "
          f"(src {src_fps:.1f}fps, every {interval}, {src_w}x{src_h}, mode={mode})")
    images = load_and_preprocess_images(
        paths, mode=mode, image_size=IMG_SIZE, patch_size=14)
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass
    try:
        os.rmdir(out_dir)
    except OSError:
        pass
    if images.dim() == 4:
        images = images.unsqueeze(0)          # [1,S,3,H,W]
    return images


def new_session(peer_id: str) -> MapSession:
    return MapSession(session_id=uuid.uuid4().hex[:12], peer_id=peer_id)
