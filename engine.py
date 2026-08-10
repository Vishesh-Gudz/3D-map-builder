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
NUM_SCALE_FRAMES = int(os.environ.get("MAP_SCALE_FRAMES", "4"))
KEYFRAME_INTERVAL = int(os.environ.get("MAP_KEYFRAME_INTERVAL", "3"))
CONF_THRESHOLD = float(os.environ.get("MAP_CONF_THRESHOLD", "1.5"))
PIXEL_STRIDE = int(os.environ.get("MAP_PIXEL_STRIDE", "4"))    # sample every Nth pixel
VOXEL_SIZE = float(os.environ.get("MAP_VOXEL_SIZE", "0.03"))   # world-unit dedupe grid
MAX_FRAMES = int(os.environ.get("MAP_MAX_FRAMES", "2000"))
MAPS_DIR = os.environ.get("MAPS_DIR", os.path.join(os.path.dirname(__file__), "maps"))

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
        camera_num_iterations=1,     # speed over marginal pose accuracy
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
    state: str = "warming"   # warming | streaming | stopped
    frames_in: int = 0
    frames_mapped: int = 0
    total_points: int = 0
    seq: int = 0
    warm_buffer: list = field(default_factory=list)
    voxels: set = field(default_factory=set)
    # accumulated cloud for the PLY (grows with dedeuped points only)
    acc_xyz: list = field(default_factory=list)
    acc_rgb: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_frame_at: float = field(default_factory=time.time)

    # ── frame ingestion ──────────────────────────────────────────────────
    def add_frame(self, jpeg_bytes: bytes) -> list["Chunk"]:
        """Feed one JPEG (HTTP debug path). Returns 0..N chunks."""
        if self.state == "stopped" or self.frames_in >= MAX_FRAMES:
            return []
        self.last_frame_at = time.time()
        self.frames_in += 1
        t0 = time.perf_counter()
        frame = _preprocess_jpeg(jpeg_bytes)
        return self._ingest(frame, t0, time.perf_counter())

    def add_frame_array(self, rgb: np.ndarray) -> list["Chunk"]:
        """Feed one decoded RGB frame (WebRTC path). Returns 0..N chunks."""
        if self.state == "stopped" or self.frames_in >= MAX_FRAMES:
            return []
        self.last_frame_at = time.time()
        self.frames_in += 1
        t0 = time.perf_counter()
        frame = _preprocess_array(rgb)
        return self._ingest(frame, t0, time.perf_counter())

    def _ingest(self, frame: torch.Tensor, t0: float, t_pre: float) -> list["Chunk"]:
        if self.state == "warming":
            self.warm_buffer.append(frame)
            if len(self.warm_buffer) < NUM_SCALE_FRAMES:
                return []
            # Phase 1 — scale block: all warm frames in one bidirectional pass.
            model = load_model()
            block = torch.stack(self.warm_buffer, dim=0).unsqueeze(0).to(_DEVICE)
            with _infer_lock, torch.no_grad(), torch.amp.autocast("cuda", dtype=_DTYPE):
                model.clean_kv_cache()
                out = model.forward(
                    block,
                    num_frame_for_scale=NUM_SCALE_FRAMES,
                    num_frame_per_block=NUM_SCALE_FRAMES,
                    causal_inference=True,
                )
                chunks = [
                    self._emit(out, block, s)
                    for s in range(NUM_SCALE_FRAMES)
                ]
            self.warm_buffer = []
            self.state = "streaming"
            return [c for c in chunks if c is not None]

        # streaming: one forward per frame, keyframe policy bounds the cache.
        model = load_model()
        stream_idx = self.frames_mapped - NUM_SCALE_FRAMES
        is_keyframe = KEYFRAME_INTERVAL <= 1 or (stream_idx % KEYFRAME_INTERVAL == 0)
        frame_b = frame.unsqueeze(0).unsqueeze(0).to(_DEVICE)  # [1,1,3,H,W]
        with _infer_lock, torch.no_grad(), torch.amp.autocast("cuda", dtype=_DTYPE):
            if not is_keyframe:
                model._set_skip_append(True)
            out = model.forward(
                frame_b,
                num_frame_for_scale=NUM_SCALE_FRAMES,
                num_frame_per_block=1,
                causal_inference=True,
            )
            if not is_keyframe:
                model._set_skip_append(False)
            t_fwd = time.perf_counter()
            chunk = self._emit(out, frame_b, 0)
        t_emit = time.perf_counter()
        print(f"[engine] pre {1000 * (t_pre - t0):.0f}ms | "
              f"fwd {1000 * (t_fwd - t_pre):.0f}ms | "
              f"emit {1000 * (t_emit - t_fwd):.0f}ms")
        return [chunk] if chunk is not None else []

    # ── point extraction ─────────────────────────────────────────────────
    def _emit(self, out: dict, images_block: torch.Tensor, s: int) -> Optional[Chunk]:
        """One frame's predictions → conf filter → stride → voxel dedupe → chunk."""
        self.frames_mapped += 1
        # GCTStream ships with enable_point=False — the official pipeline derives
        # world points by unprojecting DEPTH through the predicted pose (same as
        # demo_render's --keyframes_only_points). The model provides the helper.
        if "depth" not in out or "depth_conf" not in out:
            return None
        model = _model
        assert model is not None
        wp_t = model._unproject_depth_to_world(
            out["depth"].float(), out["pose_enc"].float()
        )                                                              # [B,S,H,W,3]
        wp = wp_t[0, s].float().cpu().numpy()                          # [H,W,3]
        conf = out["depth_conf"][0, s].float().cpu().numpy()           # [H,W]
        rgb = (
            images_block[0, s].float().cpu().numpy().transpose(1, 2, 0) * 255
        ).clip(0, 255).astype(np.uint8)                                # [H,W,3]

        st = PIXEL_STRIDE
        wp_s = wp[::st, ::st].reshape(-1, 3)
        conf_s = conf[::st, ::st].reshape(-1)
        rgb_s = rgb[::st, ::st].reshape(-1, 3)

        keep = conf_s >= CONF_THRESHOLD
        wp_s, rgb_s, conf_s = wp_s[keep], rgb_s[keep], conf_s[keep]
        if wp_s.shape[0] == 0:
            return None

        # Voxel dedupe: only voxels never seen in this session pass through —
        # re-walking an aisle adds ~nothing; map size is bounded by the SPACE.
        vox = np.floor(wp_s / VOXEL_SIZE).astype(np.int64)
        keys = vox[:, 0] * 73856093 ^ vox[:, 1] * 19349663 ^ vox[:, 2] * 83492791
        new_mask = np.fromiter(
            (k not in self.voxels for k in keys), dtype=bool, count=len(keys)
        )
        self.voxels.update(keys[new_mask].tolist())
        wp_new, rgb_new, conf_new = wp_s[new_mask], rgb_s[new_mask], conf_s[new_mask]
        if wp_new.shape[0] == 0:
            return None

        # Camera pose (c2w 4x4) for the trajectory trail.
        pose_mat = [0.0] * 16
        try:
            extri, _ = pose_encoding_to_extri_intri(
                out["pose_enc"][:, s:s + 1], images_block.shape[-2:]
            )
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
