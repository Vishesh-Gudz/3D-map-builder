#!/usr/bin/env bash
# One-shot boot for the map-builder service on a fresh GPU pod (RunPod etc).
# Idempotent — safe to re-run after a pod restart (install/fetch skip if done).
set -e

cd "$(dirname "$0")"

echo "== [1/4] python deps =="
pip install -q -e . 2>&1 | tail -1 || true
pip install -q fastapi 'uvicorn[standard]' httpx onnxruntime 2>&1 | tail -1 || true
# WebRTC subscriber: pod joins the mesh and eats the phone's real stream.
pip install -q aiortc av "python-socketio[asyncio_client]" 2>&1 | tail -1 || true

echo "== [2/4] model weights =="
python fetch_model.py

echo "== [3/4] gpu check =="
python - <<'PY'
import torch
print("  cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

echo "== [4/4] starting service on :8090 =="
# Cloud GPU (4090) fits full 518 + more scale frames + denser points.
export MAP_SCALE_FRAMES="${MAP_SCALE_FRAMES:-8}"
export MAP_PIXEL_STRIDE="${MAP_PIXEL_STRIDE:-2}"
# SERVER_URL must point at your PUBLICLY reachable Express — it is BOTH the
# chunk push target AND the socket.io signaling host the WebRTC subscriber
# joins. ROOM_ID must match the phones' room (default shipment-glasses-dev).
export ROOM_ID="${ROOM_ID:-shipment-glasses-dev}"
exec python -m uvicorn server:app --host 0.0.0.0 --port 8090
