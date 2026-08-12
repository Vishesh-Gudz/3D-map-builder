"""A/B test: run demo.py's OWN loader + inference on frames we captured live.

Point it at a folder of dumped frames (MAP_DEBUG_DUMP=1 MAP_DEBUG_EVERY=1) and
it reproduces demo.py's pipeline exactly — load_and_preprocess_images →
inference_streaming → same diagnostics engine.py prints.

    python check_offline.py debug_frames

Reading the result:
  * camera path SHORT here too  → the FOOTAGE is the problem (resolution, blur,
    motion), not our service. Fix capture, not code.
  * camera path LONG here       → our live pipeline diverges from demo somewhere
    and the bug is still ours.
"""

import os
import sys
import glob

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general

import engine


def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else "debug_frames"
    # "crop" (demo default) vs "pad" (keeps a portrait stream's full FOV).
    mode = sys.argv[2] if len(sys.argv) > 2 else "crop"
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    if not paths:
        print(f"no frames in {folder} — run a session with "
              f"MAP_DEBUG_DUMP=1 MAP_DEBUG_EVERY=1 first")
        return
    print(f"[ab] {len(paths)} frames from {folder}")
    print(f"[ab] first: {os.path.basename(paths[0])}")

    # demo.py's exact loader
    images = load_and_preprocess_images(
        paths, mode=mode, image_size=engine.IMG_SIZE, patch_size=14)
    print(f"[ab] mode={mode} preprocessed tensor {tuple(images.shape)}")

    model = engine.load_model()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=engine._DTYPE):
        preds = model.inference_streaming(
            images,
            num_scale_frames=engine.NUM_SCALE_FRAMES,
            keyframe_interval=engine.KEYFRAME_INTERVAL,
            output_device=torch.device("cpu"),
        )

    pose_enc = preds["pose_enc"]
    depth = preds["depth"]
    hw = tuple(images.shape[-2:])
    extri, _ = pose_encoding_to_extri_intri(pose_enc.float(), hw)
    c2w = torch.zeros((*extri.shape[:-2], 4, 4), dtype=extri.dtype)
    c2w[..., :3, :4] = extri
    c2w[..., 3, 3] = 1.0
    cam = c2w[0, :, :3, 3].numpy()

    path_len = float(np.linalg.norm(np.diff(cam, axis=0), axis=1).sum())
    span = float(np.linalg.norm(cam.max(0) - cam.min(0)))
    net = float(np.linalg.norm(cam[-1] - cam[0]))
    print(f"[ab] camera path {path_len:.2f} | bbox span {span:.2f} | "
          f"net displacement {net:.2f}")
    S = depth.shape[1]
    for s in (0, S // 2, S - 1):
        d = depth[0, s].float().numpy().squeeze(-1)
        print(f"[ab] frame {s}: depth mean {d.mean():.3f} std {d.std():.3f}")
    print("[ab] VERDICT: short path + tiny span => footage is the limiter "
          "(raise capture resolution); long path => live pipeline bug.")


if __name__ == "__main__":
    main()
