"""Map-builder service — HTTP wrapper around engine.py for live 3D mapping.

Run (inside this repo's venv, GPU machine):
    uvicorn server:app --host 0.0.0.0 --port 8090

API (called by shipment-glasses' Express):
    POST /session/start   {"peerId": "glasses-xxx"}          → {sessionId, state}
    POST /session/{id}/frame   body = raw JPEG bytes         → {accepted, state, chunks}
    POST /session/{id}/stop                                  → final stats + PLY path
    GET  /session/{id}/chunks?since=<seq>                    → pull mode (testing / no push sink)
    GET  /health

Chunk delivery: if SERVER_URL is set, each chunk is PUSHED as
POST {SERVER_URL}/map-chunk (Express relays to the viewer over Socket.IO).
Chunks are ALSO buffered for pull — harmless duplication guard lives in the
viewer (seq-keyed).
"""

# Allocator must be configured before torch loads (see demo.py rationale).
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import asyncio
import base64
import threading
import time
from typing import Optional

import glob

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import engine
from rtc_client import RtcSubscriber

SERVER_URL = os.environ.get("SERVER_URL", "")  # e.g. http://localhost:3001
IDLE_TIMEOUT_SEC = int(os.environ.get("MAP_IDLE_TIMEOUT", "60"))
CHUNK_BUFFER_MAX = 500
# WebRTC subscriber: the pod joins the signaling room (same Express host as
# SERVER_URL) and consumes the phone's real video stream during a session.
ROOM_ID = os.environ.get("ROOM_ID", "shipment-glasses-dev")
RTC_PEER_ID = os.environ.get("RTC_PEER_ID", "viewer-mapper")  # "viewer" prefix → phones auto-offer
STUN_URLS = [u for u in os.environ.get(
    "STUN_URLS", "stun:stun.l.google.com:19302").split(",") if u]

UPLOAD_DIR = os.environ.get("UPLOAD_DIR",
                            os.path.join(os.path.dirname(__file__), "uploads"))
VIDEO_FPS = int(os.environ.get("MAP_VIDEO_FPS", "10"))  # demo.py's sampling rate

app = FastAPI(title="sg-map-builder")
# The browser uploads videos straight to the pod (bypassing Express — the file
# is far too big to relay) and polls for chunks, so it needs CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_session: Optional[engine.MapSession] = None
_chunk_buffer: list[dict] = []
_push_client = httpx.AsyncClient(timeout=10)
# Chunks leave through a background worker so the capture loop returns to the
# GPU the moment inference is done — the pod→Express push must not hold up the
# next frame.
_push_queue: "asyncio.Queue[dict]" = asyncio.Queue(maxsize=200)
_rtc: Optional[RtcSubscriber] = None
_capture_task: Optional[asyncio.Task] = None


def _enqueue_chunks(s: engine.MapSession, chunks: list) -> int:
    """Buffer + queue chunk payloads for background delivery. Returns new pts."""
    payloads = [_chunk_payload(s, c) for c in chunks]
    for p in payloads:
        _chunk_buffer.append(p)
        if len(_chunk_buffer) > CHUNK_BUFFER_MAX:
            _chunk_buffer.pop(0)
        try:
            _push_queue.put_nowait(p)
        except asyncio.QueueFull:
            print(f"[server] push queue full — dropping seq {p['seq']}")
    return sum(c.count for c in chunks)


def _chunk_payload(session: engine.MapSession, chunk: engine.Chunk) -> dict:
    return {
        "sessionId": session.session_id,
        "peerId": session.peer_id,
        "seq": chunk.seq,
        "count": chunk.count,
        "points": base64.b64encode(chunk.points).decode("ascii"),
        "colors": base64.b64encode(chunk.colors).decode("ascii"),
        "confs": base64.b64encode(chunk.confs).decode("ascii"),
        "pose": chunk.pose,
    }


async def _push_chunk(payload: dict) -> None:
    if not SERVER_URL:
        return
    try:
        await _push_client.post(f"{SERVER_URL.rstrip('/')}/map-chunk", json=payload)
    except Exception as err:  # push failures never break the session
        print(f"[server] chunk push failed: {err}")


async def _push_worker() -> None:
    """Drain the chunk queue in order — one slow push never blocks inference."""
    while True:
        payload = await _push_queue.get()
        t0 = time.perf_counter()
        await _push_chunk(payload)
        dt = (time.perf_counter() - t0) * 1000
        if dt > 500:
            print(f"[push] seq {payload.get('seq')} took {dt:.0f}ms "
                  f"(queue {_push_queue.qsize()})")


CAPTURE_FPS = float(os.environ.get("MAP_CAPTURE_FPS", "10"))  # demo samples video @10fps


async def _capture_loop(s: engine.MapSession) -> None:
    """Buffer the WebRTC stream at a steady rate (demo.py samples video @10fps).
    We keep EVERY sampled frame — the whole clip is reconstructed together on
    stop via inference_streaming, so temporal continuity (what the model needs
    for pose) is preserved instead of grabbing scattered 'freshest' frames."""
    assert _rtc is not None
    min_dt = 1.0 / CAPTURE_FPS if CAPTURE_FPS > 0 else 0.0
    last_ts = 0.0
    last_buf = 0.0
    last_status = 0.0
    started = time.time()
    print(f"[capture] buffering {s.peer_id} @ {CAPTURE_FPS}fps — waiting for stream")
    while _session is s and s.state == "capturing":
        slot = _rtc.bus.get(s.peer_id)
        now = time.time()
        if slot is None or slot[1] <= last_ts or (now - last_buf) < min_dt:
            if last_ts == 0.0 and now - started > 30:
                print("[capture] no frames after 30s — phone streaming? NAT ok?")
                started = now
            await asyncio.sleep(0.005)
            continue
        arr, last_ts = slot
        last_buf = now
        await asyncio.to_thread(s.add_frame_array, arr)
        if now - last_status >= 1.0:
            last_status = now
            try:
                _push_queue.put_nowait({
                    "sessionId": s.session_id, "peerId": s.peer_id,
                    "status": {"state": "warming", "frames": s.frames_in,
                               "points": 0},
                })
            except asyncio.QueueFull:
                pass
    print(f"[capture] buffered {s.frames_in} frames for {s.peer_id}")


async def _teardown_rtc() -> None:
    """Stop the capture loop and leave the room (frees the phone's uplink)."""
    global _capture_task
    if _capture_task is not None:
        _capture_task.cancel()
        try:
            await _capture_task
        except (asyncio.CancelledError, Exception):
            pass
        _capture_task = None
    if _rtc is not None:
        await _rtc.stop()


@app.on_event("startup")
async def _boot() -> None:
    # Load the model up front so the first session doesn't eat the delay.
    await asyncio.to_thread(engine.load_model)
    asyncio.get_event_loop().create_task(_idle_reaper())
    asyncio.get_event_loop().create_task(_push_worker())


async def _idle_reaper() -> None:
    """Auto-stop a session whose camera went away — reconstruct what we buffered
    (only while still capturing; a session already processing is left alone)."""
    while True:
        await asyncio.sleep(10)
        s = _session
        if (s and s.state == "capturing"
                and time.time() - s.last_frame_at > IDLE_TIMEOUT_SEC):
            print(f"[server] session {s.session_id} idle → reconstructing "
                  f"{s.frames_in} frames")
            s.state = "processing"
            await _teardown_rtc()
            asyncio.create_task(_reconstruct_and_finish(s))


@app.get("/health")
async def health() -> dict:
    s = _session
    return {
        "status": "ok",
        "model": "loaded" if engine._model is not None else "loading",
        "session": None if s is None else {
            "sessionId": s.session_id, "peerId": s.peer_id, "state": s.state,
            "framesBuffered": s.frames_in, "framesMapped": s.frames_mapped,
            "points": s.total_points,
        },
    }


@app.post("/session/start")
async def start(body: dict) -> dict:
    global _session, _chunk_buffer, _rtc, _capture_task
    peer_id = str(body.get("peerId", "unknown"))
    if _session is not None and _session.state != "stopped":
        # One live session (Phase A) — restart replaces it.
        await asyncio.to_thread(_session.finalize)
    await _teardown_rtc()
    _session = engine.new_session(peer_id)
    _chunk_buffer = []
    if engine.DEBUG_DUMP:  # fresh gallery per session
        for f in glob.glob(os.path.join(engine.DEBUG_DIR, "*.jpg")):
            try:
                os.remove(f)
            except OSError:
                pass

    # Join the WebRTC room and consume the phone's real stream. Falls back to
    # the HTTP /frame path (old JPEG lane) if signaling is unreachable.
    if SERVER_URL:
        if _rtc is None:
            _rtc = RtcSubscriber(SERVER_URL, ROOM_ID, RTC_PEER_ID, STUN_URLS)
        try:
            await _rtc.start()
            _capture_task = asyncio.create_task(_capture_loop(_session))
        except Exception as err:
            print(f"[server] rtc connect failed ({err}) — HTTP frame path only")

    print(f"[server] session {_session.session_id} started for {peer_id}")
    return {"sessionId": _session.session_id, "state": _session.state}


@app.post("/session/{session_id}/frame")
async def frame(session_id: str, request: Request) -> dict:
    s = _session
    if s is None or s.session_id != session_id:
        raise HTTPException(404, "no such session")
    if s.state == "stopped":
        raise HTTPException(409, "session stopped")

    t0 = time.perf_counter()
    jpeg = await request.body()
    t_recv = time.perf_counter()
    if len(jpeg) < 100:
        raise HTTPException(400, "empty frame")

    chunks = await asyncio.to_thread(s.add_frame, jpeg)
    t_infer = time.perf_counter()

    new_pts = _enqueue_chunks(s, chunks)
    print(f"[frame {s.frames_mapped}] recv {1000 * (t_recv - t0):.0f}ms | "
          f"infer {1000 * (t_infer - t_recv):.0f}ms | +{new_pts} pts | "
          f"queue {_push_queue.qsize()}")

    return {
        "accepted": True,
        "state": s.state,
        "framesMapped": s.frames_mapped,
        "points": s.total_points,
        "chunks": len(chunks),
    }


@app.get("/session/{session_id}/chunks")
async def chunks(session_id: str, since: int = 0) -> JSONResponse:
    s = _session
    if s is None or s.session_id != session_id:
        raise HTTPException(404, "no such session")
    pending = [c for c in _chunk_buffer if c["seq"] > since]
    return JSONResponse({"chunks": pending, "state": s.state})


@app.get("/debug/frames", response_class=HTMLResponse)
async def debug_frames() -> HTMLResponse:
    """Browser gallery of the raw frames the model received (MAP_DEBUG_DUMP=1).
    The filename carries the real capture resolution (…_WxH.jpg)."""
    files = sorted(os.path.basename(p) for p in glob.glob(
        os.path.join(engine.DEBUG_DIR, "*.jpg")))
    if not files:
        return HTMLResponse(
            "<p style='font-family:sans-serif'>No debug frames yet. Set "
            "<code>MAP_DEBUG_DUMP=1</code> before start.sh, then run a map "
            "session.</p>")
    cells = "".join(
        f'<figure style="margin:0"><img src="/debug/frames/{f}" '
        f'style="width:100%;display:block"><figcaption '
        f'style="font:12px monospace;color:#333;padding:4px">{f}</figcaption>'
        f'</figure>' for f in files)
    html = (
        "<html><body style='background:#111;margin:0'>"
        f"<p style='color:#ccc;font:13px sans-serif;padding:8px'>"
        f"{len(files)} frames — filename shows real capture resolution</p>"
        "<div style='display:grid;gap:8px;padding:8px;"
        "grid-template-columns:repeat(auto-fill,minmax(280px,1fr))'>"
        f"{cells}</div></body></html>")
    return HTMLResponse(html)


@app.get("/debug/frames/{name}")
async def debug_frame(name: str) -> FileResponse:
    if "/" in name or "\\" in name or not name.endswith(".jpg"):
        raise HTTPException(404, "bad name")
    path = os.path.join(engine.DEBUG_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(404, "no such frame")
    return FileResponse(path, media_type="image/jpeg")


# ── Video upload path ────────────────────────────────────────────────────────
# Self-contained alternative to live WebRTC mapping: upload a clip, the pod
# reconstructs it with demo.py's exact pipeline, the browser polls the chunks
# and renders them in the same three.js pane. Full-resolution input, so none of
# the live path's bandwidth/NAT constraints apply.

_jobs: dict[str, dict] = {}


def _run_video_job(job_id: str, path: str) -> None:
    """Decode → reconstruct → chunks. Runs in a worker thread."""
    job = _jobs[job_id]
    session = engine.MapSession(session_id=job_id, peer_id="upload")
    try:
        job["state"] = "decoding"
        images = engine.load_video_frames(path, fps=VIDEO_FPS)
        job["frames"] = int(images.shape[1])
        job["state"] = "reconstructing"
        chunks = session.process(images=images)
        job["state"] = "extracting"
        job["chunks"] = [_chunk_payload(session, c) for c in chunks]
        job["stats"] = session.finalize()
        job["points"] = session.total_points
        job["state"] = "done"
        print(f"[video] job {job_id} done: {job['points']} points")
    except Exception as err:
        import traceback
        traceback.print_exc()
        job["state"] = "error"
        job["error"] = str(err)[:300]
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.post("/video/upload")
async def video_upload(file: UploadFile = File(...)) -> dict:
    if engine._model is None:
        raise HTTPException(503, "model still loading")
    job_id = engine.uuid.uuid4().hex[:12]
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1][:8] or ".mp4"
    path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    size = 0
    with open(path, "wb") as f:
        while True:
            block = await file.read(1 << 20)
            if not block:
                break
            f.write(block)
            size += len(block)
    print(f"[video] job {job_id} received {size / 1e6:.1f} MB ({file.filename})")
    _jobs[job_id] = {"state": "queued", "frames": 0, "points": 0,
                     "chunks": [], "stats": None, "error": None}
    # One GPU, one job at a time — the model lock inside the engine serialises
    # anyway; a thread keeps the event loop free to serve status polls.
    threading.Thread(target=_run_video_job, args=(job_id, path), daemon=True).start()
    return {"jobId": job_id, "sizeBytes": size}


@app.get("/video/{job_id}/status")
async def video_status(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return {
        "jobId": job_id, "state": job["state"], "frames": job["frames"],
        "points": job["points"], "chunks": len(job["chunks"]),
        "error": job["error"], "stats": job["stats"],
        # Fusion snaps every point to this grid, so it IS the point spacing —
        # the viewer needs it to size splats. Guessing from the bounding box
        # gets it wrong: spacing is fixed, only the point COUNT grows with the
        # map, so extent-scaled sizing over-draws on big scenes.
        "voxelSize": engine.VOXEL_SIZE,
    }


# 2MB pages. 6MB still drew ERR_HTTP2_PROTOCOL_ERROR through the RunPod proxy;
# smaller responses transfer reliably and a dropped one costs little to retry.
CHUNK_PAGE_BYTES = int(os.environ.get("MAP_CHUNK_PAGE_BYTES", str(2 * 1024 * 1024)))


@app.get("/video/{job_id}/chunks")
async def video_chunks(job_id: str, since: int = 0, limit: int = 40) -> JSONResponse:
    """Paged chunk pull — the browser appends each page to the point cloud.

    Paged by BYTES as well as count: fused point chunks are ~1-5 MB of base64
    each, so honouring `limit` alone built responses in the hundreds of MB and
    RunPod's proxy aborted them with ERR_HTTP2_PROTOCOL_ERROR (the map simply
    never appeared). Always return at least one chunk so a single oversized one
    can still make progress rather than deadlocking the client.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    all_chunks = job["chunks"]
    page: list = []
    budget = CHUNK_PAGE_BYTES
    for c in all_chunks[since:since + max(1, limit)]:
        size = len(c.get("points", "")) + len(c.get("colors", "")) + len(c.get("confs", ""))
        if page and size > budget:
            break
        page.append(c)
        budget -= size
    return JSONResponse({
        "state": job["state"],
        "total": len(all_chunks),
        "next": since + len(page),
        "chunks": page,
    })


@app.get("/maps/{session_id}.ply")
async def download_ply(session_id: str) -> FileResponse:
    """Serve a finished session's point cloud (works after the session ends)."""
    path = os.path.join(engine.MAPS_DIR, f"{session_id}.ply")
    if not os.path.isfile(path) or "/" in session_id or "\\" in session_id:
        raise HTTPException(404, "no such map")
    return FileResponse(path, media_type="application/octet-stream",
                        filename=f"map-{session_id}.ply")


async def _reconstruct_and_finish(s: engine.MapSession) -> None:
    """Heavy pass: run inference_streaming over the buffered clip, stream the
    resulting chunks to Express (in order), write the PLY, then push a {done}
    message — Express relays chunks while `active` is alive and turns {done}
    into map-done. Runs in the background so the /stop HTTP returns fast."""
    global _session
    try:
        chunks = await asyncio.to_thread(s.process)
        for c in chunks:
            payload = _chunk_payload(s, c)
            _chunk_buffer.append(payload)
            if len(_chunk_buffer) > CHUNK_BUFFER_MAX:
                _chunk_buffer.pop(0)
            await _push_chunk(payload)  # ordered, synchronous — all land before done
        stats = await asyncio.to_thread(s.finalize)
    except Exception as err:
        import traceback
        traceback.print_exc()
        print(f"[server] reconstruction failed: {err}")
        stats = {"sessionId": s.session_id, "peerId": s.peer_id,
                 "frames": s.frames_in, "points": s.total_points,
                 "durationSec": round(time.time() - s.created_at, 1), "ply": ""}
    if SERVER_URL:
        try:
            await _push_client.post(
                f"{SERVER_URL.rstrip('/')}/map-chunk",
                json={"sessionId": stats["sessionId"], "peerId": stats["peerId"],
                      "done": True, "stats": stats})
        except Exception:
            pass
    if _session is s:
        _session = None
    print(f"[server] session {s.session_id} done: {stats}")


@app.post("/session/{session_id}/stop")
async def stop(session_id: str) -> dict:
    s = _session
    if s is None or s.session_id != session_id:
        raise HTTPException(404, "no such session")
    if s.state != "capturing":
        return {"accepted": True, "state": s.state, "frames": s.frames_in}
    s.state = "processing"
    await _teardown_rtc()  # stop buffering + free the phone's uplink
    asyncio.create_task(_reconstruct_and_finish(s))
    print(f"[server] session {session_id} → reconstructing {s.frames_in} frames")
    return {"accepted": True, "state": "processing", "frames": s.frames_in}
