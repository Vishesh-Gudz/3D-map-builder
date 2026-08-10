# Deploy the map-builder service on RunPod (GPU)

The live 3D-mapping GPU service behind shipment-glasses' "Build 3D Map" button.
Frames in over HTTP → point-cloud chunks pushed back to the app's Express.

```
phone → Express (/detect tee) → THIS service (RunPod 4090) → chunks → Express → viewer 3D Map
```

## 1. Create the pod
- GPU: **RTX 4090** (24 GB, ~$0.74/hr) — model needs ~3 GB, runs full 518 res ~20 fps
- Template: **PyTorch 2.8 / CUDA 12.8** (or any CUDA 12.8 base with torch 2.8)
- **Network volume** (~30 GB) mounted at `/workspace` — the 4.6 GB weights download
  ONCE and survive restarts
- **Expose HTTP port 8090** → RunPod gives `https://<pod-id>-8090.proxy.runpod.net`

## 2. Run the service (pod web terminal)
```bash
cd /workspace
git clone https://github.com/Vishesh-Gudz/3D-map-builder app
cd app
# SERVER_URL = your PUBLICLY reachable Express (chunk push target)
export SERVER_URL=https://sg-0sab.onrender.com
bash start.sh
```
First run: downloads weights (~4.6 GB) + loads model (~30 s). Leave it running.

## 3. Verify
```bash
curl https://<pod-id>-8090.proxy.runpod.net/health
# → {"status":"ok","model":"loaded", ...}
```

## 4. Point the app at it
In shipment-glasses' Express env (`apps/server/.env` locally, or the Render
dashboard):
```
MAPPER_URL=https://<pod-id>-8090.proxy.runpod.net
```
Restart Express. Phone/viewer unchanged — "Build 3D Map" now runs on the 4090.

## Tunables (env, set before `start.sh`)
| var | default | effect |
|-----|---------|--------|
| `MAP_SCALE_FRAMES` | 8 | warmup frames for scale bootstrap |
| `MAP_PIXEL_STRIDE` | 2 | 1 = densest (4× points), 4 = sparse/fast |
| `MAP_KEYFRAME_INTERVAL` | 3 | KV-cache keyframe spacing (long sessions) |
| `MAP_CONF_THRESHOLD` | 1.5 | drop low-confidence points |
| `MAP_VOXEL_SIZE` | 0.03 | dedupe grid (world units) |
| `ROOM_ID` | shipment-glasses-dev | signaling room the WebRTC subscriber joins |
| `RTC_PEER_ID` | viewer-mapper | pod's peer id ("viewer" prefix → phones offer) |
| `STUN_URLS` | stun:stun.l.google.com:19302 | comma-separated ICE servers |

## Frame path
During a session the pod joins the WebRTC room as one more viewer and decodes
the phone's REAL video stream (sharp, 30fps available) — the HTTP `/frame`
endpoint remains as a debug fallback. The pod leaves the room on stop, so the
phone only pays the extra uplink leg while actually mapping.

## ⚠️ Cost
RunPod bills per running hour. **STOP the pod when not mapping.** The network
volume keeps the weights cached (tiny storage fee) so restarts are fast.
