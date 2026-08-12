#!/usr/bin/env bash
# One-shot boot for the map-builder service on a fresh GPU pod (RunPod etc).
# Idempotent — safe to re-run after a pod restart (install/fetch skip if done).
set -e

cd "$(dirname "$0")"

echo "== [1/4] python deps =="
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -q --ignore-installed -e .
pip install -q --ignore-installed fastapi 'uvicorn[standard]' httpx onnxruntime aiortc av "python-socketio[asyncio_client]" python-multipart opencv-python-headless

echo "== [2/4] model weights =="
python fetch_model.py

echo "== [3/4] gpu check =="
python - <<'PY'
import torch
print("  cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

echo "== [4/4] starting service on :8090 =="
# Quality-relevant knobs live in engine.py so a plain `bash start.sh` reproduces
# the measured-good config — do NOT re-default them here, or results silently
# depend on which shell you started from.
export MAP_SCALE_FRAMES="${MAP_SCALE_FRAMES:-8}"
# SERVER_URL must point at your PUBLICLY reachable Express — it is BOTH the
# chunk push target AND the socket.io signaling host the WebRTC subscriber
# joins. ROOM_ID must match the phones' room (default shipment-glasses-dev).
export ROOM_ID="${ROOM_ID:-shipment-glasses-dev}"
exec python -m uvicorn server:app --host 0.0.0.0 --port 8090
