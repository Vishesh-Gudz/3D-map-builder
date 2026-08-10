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

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

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

app = FastAPI(title="sg-map-builder")

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


async def _capture_loop(s: engine.MapSession) -> None:
    """WebRTC frame path: pull the freshest decoded frame at GPU pace. The bus
    always holds the latest video frame, so the model never eats a backlog —
    it samples the stream exactly as fast as it can infer (~7fps on a 4090)."""
    assert _rtc is not None
    last_ts = 0.0
    last_status = 0.0
    started = time.time()
    print(f"[capture] loop up for {s.peer_id} — waiting for stream")
    while _session is s and s.state != "stopped":
        slot = _rtc.bus.get(s.peer_id)
        if slot is None or slot[1] <= last_ts:
            if last_ts == 0.0 and time.time() - started > 30:
                print("[capture] no frames after 30s — phone streaming? NAT ok?")
                started = time.time()
            await asyncio.sleep(0.02)
            continue
        arr, last_ts = slot
        t0 = time.perf_counter()
        chunks = await asyncio.to_thread(s.add_frame_array, arr)
        new_pts = _enqueue_chunks(s, chunks)
        # Express no longer sees frames (no tee), so the viewer's frame/point
        # counters ride the push channel as throttled status messages.
        if time.time() - last_status >= 1.0:
            last_status = time.time()
            try:
                _push_queue.put_nowait({
                    "sessionId": s.session_id, "peerId": s.peer_id,
                    "status": {"state": s.state, "frames": s.frames_mapped,
                               "points": s.total_points},
                })
            except asyncio.QueueFull:
                pass
        if new_pts:
            print(f"[capture {s.frames_mapped}] "
                  f"infer {1000 * (time.perf_counter() - t0):.0f}ms | "
                  f"+{new_pts} pts | queue {_push_queue.qsize()}")
    print(f"[capture] loop done for {s.peer_id} ({s.frames_mapped} frames)")


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
    """Auto-stop a session whose camera went away (no frames for a while)."""
    global _session
    while True:
        await asyncio.sleep(10)
        s = _session
        if s and s.state != "stopped" and time.time() - s.last_frame_at > IDLE_TIMEOUT_SEC:
            print(f"[server] session {s.session_id} idle → auto-stop")
            stats = await asyncio.to_thread(s.finalize)
            _session = None
            await _teardown_rtc()
            if SERVER_URL:
                try:
                    await _push_client.post(
                        f"{SERVER_URL.rstrip('/')}/map-chunk",
                        json={"sessionId": stats["sessionId"], "peerId": stats["peerId"],
                              "done": True, "stats": stats},
                    )
                except Exception:
                    pass


@app.get("/health")
async def health() -> dict:
    s = _session
    return {
        "status": "ok",
        "model": "loaded" if engine._model is not None else "loading",
        "session": None if s is None else {
            "sessionId": s.session_id, "peerId": s.peer_id, "state": s.state,
            "frames": s.frames_mapped, "points": s.total_points,
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


@app.get("/maps/{session_id}.ply")
async def download_ply(session_id: str) -> FileResponse:
    """Serve a finished session's point cloud (works after the session ends)."""
    path = os.path.join(engine.MAPS_DIR, f"{session_id}.ply")
    if not os.path.isfile(path) or "/" in session_id or "\\" in session_id:
        raise HTTPException(404, "no such map")
    return FileResponse(path, media_type="application/octet-stream",
                        filename=f"map-{session_id}.ply")


@app.post("/session/{session_id}/stop")
async def stop(session_id: str) -> dict:
    global _session
    s = _session
    if s is None or s.session_id != session_id:
        raise HTTPException(404, "no such session")
    stats = await asyncio.to_thread(s.finalize)
    _session = None
    await _teardown_rtc()
    print(f"[server] session {session_id} stopped: {stats}")
    return stats
