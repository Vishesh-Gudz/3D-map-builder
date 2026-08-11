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
from PIL import Image
from torchvision.transforms.functional import to_tensor

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

# ── Config (env) ─────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "lingbot-map.pt"))
# MUST stay 518 — the checkpoint/architecture is fixed-size (no pos-embed
# interpolation; smaller inputs break internal token reshapes). VRAM control
# comes from bf16 + kv sliding window + camera_iters=1 instead.
IMG_SIZE = int(os.environ.get("MAP_IMG_SIZE", "518"))
NUM_SCALE_FRAMES = int(os.environ.get("MAP_SCALE_FRAMES", "8"))
# 1 = every frame anchored in the KV cache (demo.py's value for <320-frame
# runs). Higher trades pose accuracy for cache memory — do NOT raise for
# quality-sensitive maps.
KEYFRAME_INTERVAL = int(os.environ.get("MAP_KEYFRAME_INTERVAL", "1"))
CAMERA_ITERS = int(os.environ.get("MAP_CAMERA_ITERS", "4"))  # demo default; 1 = fast/inaccurate
CONF_THRESHOLD = float(os.environ.get("MAP_CONF_THRESHOLD", "1.5"))
PIXEL_STRIDE = int(os.environ.get("MAP_PIXEL_STRIDE", "4"))    # sample every Nth pixel
VOXEL_SIZE = float(os.environ.get("MAP_VOXEL_SIZE", "0.03"))   # world-unit dedupe grid
MAX_FRAMES = int(os.environ.get("MAP_MAX_FRAMES", "2000"))
MAPS_DIR = os.environ.get("MAPS_DIR", os.path.join(os.path.dirname(__file__), "maps"))
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
    """JPEG bytes → canonical-crop tensor [3, H, W] — the EXACT preprocessing
    demo.py uses (via a temp file, so resize/crop stays identical)."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(jpeg_bytes)
        path = f.name
    try:
        images = load_and_preprocess_images([path], mode="crop",
                                            image_size=IMG_SIZE, patch_size=14)
        return images[0]
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _preprocess_array(rgb: np.ndarray) -> torch.Tensor:
    """Decoded RGB frame [H,W,3] → canonical-crop tensor. Mirrors
    load_and_preprocess_images' crop mode exactly (resize width→IMG_SIZE,
    height to /14 multiple, center-crop overflow) — no JPEG, no disk."""
    img = Image.fromarray(rgb)
    w, h = img.size
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
    def process(self) -> list["Chunk"]:
        """Run the OFFICIAL streaming reconstruction over the buffered clip,
        exactly like demo.py: Phase-1 scale block, then frame-by-frame with the
        KV cache and full camera-pose refinement. Returns per-frame chunks for
        the viewer. This is where geometry is actually built."""
        if not self.frames_buf:
            return []
        self.state = "processing"
        model = load_model()
        # Pad any odd-aspect frame to the modal shape so torch.stack succeeds
        # (phone is portrait → nearly all frames are already 518x518).
        shapes = {tuple(f.shape) for f in self.frames_buf}
        if len(shapes) > 1:
            h = max(f.shape[1] for f in self.frames_buf)
            w = max(f.shape[2] for f in self.frames_buf)
            self.frames_buf = [
                torch.nn.functional.pad(
                    f, (0, w - f.shape[2], 0, h - f.shape[1]), value=1.0)
                for f in self.frames_buf
            ]
        images = torch.stack(self.frames_buf, dim=0).unsqueeze(0)  # [1,S,3,H,W] CPU
        self.frames_buf = []
        S = images.shape[1]
        t0 = time.time()
        print(f"[engine] reconstructing {S} frames (inference_streaming, "
              f"scale={NUM_SCALE_FRAMES}, keyframe={KEYFRAME_INTERVAL}, "
              f"cam_iters={CAMERA_ITERS})…")
        with _infer_lock, torch.no_grad(), torch.amp.autocast("cuda", dtype=_DTYPE):
            preds = model.inference_streaming(
                images,
                num_scale_frames=NUM_SCALE_FRAMES,
                keyframe_interval=KEYFRAME_INTERVAL,
                output_device=torch.device("cpu"),
            )
        pose_enc = preds["pose_enc"]        # [1,S,9]
        depth = preds["depth"]              # [1,S,H,W,1]
        depth_conf = preds["depth_conf"]    # [1,S,H,W]
        print(f"[engine] inference done in {time.time() - t0:.1f}s — extracting")

        chunks: list[Chunk] = []
        for s in range(S):
            # Unproject one frame at a time to bound peak memory.
            with _infer_lock, torch.no_grad():
                wp_t = model._unproject_depth_to_world(
                    depth[:, s:s + 1].float().to(_DEVICE),
                    pose_enc[:, s:s + 1].float().to(_DEVICE),
                )
            wp = wp_t[0, 0].float().cpu().numpy()                       # [H,W,3]
            conf = depth_conf[0, s].float().numpy()                    # [H,W]
            rgb = (images[0, s].float().numpy().transpose(1, 2, 0) * 255
                   ).clip(0, 255).astype(np.uint8)                     # [H,W,3]
            chunk = self._points_to_chunk(wp, conf, rgb, pose_enc[:, s:s + 1])
            if chunk is not None:
                chunks.append(chunk)
        print(f"[engine] extracted {self.total_points} points in {len(chunks)} chunks")
        return chunks

    def _points_to_chunk(
        self, wp: np.ndarray, conf: np.ndarray, rgb: np.ndarray, pose_enc_f,
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

        pose_mat = [0.0] * 16
        try:
            extri, _ = pose_encoding_to_extri_intri(pose_enc_f, (wp.shape[0], wp.shape[1]))
            e = np.eye(4, dtype=np.float32)
            e[:3, :4] = extri[0, 0].float().cpu().numpy()
            pose_mat = np.linalg.inv(e).reshape(-1).tolist()  # w2c → c2w
        except Exception:
            pass

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


def new_session(peer_id: str) -> MapSession:
    return MapSession(session_id=uuid.uuid4().hex[:12], peer_id=peer_id)
