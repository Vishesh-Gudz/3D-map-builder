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
from fastapi.responses import JSONResponse

import engine

SERVER_URL = os.environ.get("SERVER_URL", "")  # e.g. http://localhost:3001
IDLE_TIMEOUT_SEC = int(os.environ.get("MAP_IDLE_TIMEOUT", "60"))
CHUNK_BUFFER_MAX = 500

app = FastAPI(title="sg-map-builder")

_session: Optional[engine.MapSession] = None
_chunk_buffer: list[dict] = []
_push_client = httpx.AsyncClient(timeout=10)


def _chunk_payload(session: engine.MapSession, chunk: engine.Chunk) -> dict:
    return {
        "sessionId": session.session_id,
        "peerId": session.peer_id,
        "seq": chunk.seq,
        "count": chunk.count,
        "points": base64.b64encode(chunk.points).decode("ascii"),
        "colors": base64.b64encode(chunk.colors).decode("ascii"),
        "pose": chunk.pose,
    }


async def _push_chunk(payload: dict) -> None:
    if not SERVER_URL:
        return
    try:
        await _push_client.post(f"{SERVER_URL.rstrip('/')}/map-chunk", json=payload)
    except Exception as err:  # push failures never break the session
        print(f"[server] chunk push failed: {err}")


@app.on_event("startup")
async def _boot() -> None:
    # Load the model up front so the first session doesn't eat the delay.
    await asyncio.to_thread(engine.load_model)
    asyncio.get_event_loop().create_task(_idle_reaper())


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
    global _session, _chunk_buffer
    peer_id = str(body.get("peerId", "unknown"))
    if _session is not None and _session.state != "stopped":
        # One live session (Phase A) — restart replaces it.
        await asyncio.to_thread(_session.finalize)
    _session = engine.new_session(peer_id)
    _chunk_buffer = []
    print(f"[server] session {_session.session_id} started for {peer_id}")
    return {"sessionId": _session.session_id, "state": _session.state}


@app.post("/session/{session_id}/frame")
async def frame(session_id: str, request: Request) -> dict:
    s = _session
    if s is None or s.session_id != session_id:
        raise HTTPException(404, "no such session")
    if s.state == "stopped":
        raise HTTPException(409, "session stopped")

    jpeg = await request.body()
    if len(jpeg) < 100:
        raise HTTPException(400, "empty frame")

    chunks = await asyncio.to_thread(s.add_frame, jpeg)

    payloads = [_chunk_payload(s, c) for c in chunks]
    for p in payloads:
        _chunk_buffer.append(p)
        if len(_chunk_buffer) > CHUNK_BUFFER_MAX:
            _chunk_buffer.pop(0)
        await _push_chunk(p)

    return {
        "accepted": True,
        "state": s.state,
        "framesMapped": s.frames_mapped,
        "points": s.total_points,
        "chunks": len(payloads),
    }


@app.get("/session/{session_id}/chunks")
async def chunks(session_id: str, since: int = 0) -> JSONResponse:
    s = _session
    if s is None or s.session_id != session_id:
        raise HTTPException(404, "no such session")
    pending = [c for c in _chunk_buffer if c["seq"] > since]
    return JSONResponse({"chunks": pending, "state": s.state})


@app.post("/session/{session_id}/stop")
async def stop(session_id: str) -> dict:
    global _session
    s = _session
    if s is None or s.session_id != session_id:
        raise HTTPException(404, "no such session")
    stats = await asyncio.to_thread(s.finalize)
    _session = None
    print(f"[server] session {session_id} stopped: {stats}")
    return stats
