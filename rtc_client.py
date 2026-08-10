"""WebRTC subscriber for the map-builder — the pod joins the shipment-glasses
mesh as one more "viewer" and consumes the phone's REAL video stream.

Why: the HTTP-JPEG lane delivered ~1 blurry 640px frame/sec — this model wants
dense, sharp video (that's what the offline demo eats). Here the pod speaks the
same socket.io signaling protocol as the Next.js viewer (publishers auto-offer
to any peer id starting with "viewer"), answers with aiortc, and decodes the
track in-process. The mapping loop then pulls the freshest frame at GPU pace.

Lifecycle: connect() on session start, disconnect() on stop — the phone only
pays the extra mesh uplink leg WHILE mapping.

The FrameBus is deliberately dumb (peer id -> latest ndarray): swapping the
mesh for an SFU (MediaMTX/WHEP) later only replaces this file's connection
code, never the mapping loop.
"""

import asyncio
import os
import time
from typing import Optional

import socketio
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.sdp import candidate_from_sdp

# TURN relay — REQUIRED in practice: the phone and the pod both sit behind NAT
# (home WiFi + RunPod container) and plain STUN hole-punch fails. TURN on THIS
# side alone is enough: the phone can always reach the TURN server's public
# address, so media flows phone↔TURN↔pod. Defaults are the free OpenRelay
# service; override with your own coturn for production.
TURN_URLS = [u for u in os.environ.get(
    "TURN_URLS",
    "turn:openrelay.metered.ca:80,turn:openrelay.metered.ca:443",
).split(",") if u]
TURN_USERNAME = os.environ.get("TURN_USERNAME", "openrelayproject")
TURN_PASSWORD = os.environ.get("TURN_PASSWORD", "openrelayproject")


class FrameBus:
    """Latest decoded frame per publisher. Overwrite-only — the mapper always
    wants the freshest view, never a backlog."""

    def __init__(self) -> None:
        self._slots: dict[str, tuple] = {}  # peer_id -> (rgb ndarray, monotonic ts)

    def put(self, peer_id: str, arr) -> None:
        self._slots[peer_id] = (arr, time.monotonic())

    def get(self, peer_id: str) -> Optional[tuple]:
        return self._slots.get(peer_id)

    def clear(self) -> None:
        self._slots.clear()


class RtcSubscriber:
    def __init__(
        self,
        url: str,
        room_id: str,
        peer_id: str = "viewer-mapper",
        stun_urls: Optional[list] = None,
    ) -> None:
        self.url = url
        self.room_id = room_id
        self.peer_id = peer_id
        self.stun_urls = stun_urls or ["stun:stun.l.google.com:19302"]
        self.bus = FrameBus()
        self._pcs: dict[str, RTCPeerConnection] = {}
        self._connected = False
        self.sio = socketio.AsyncClient(reconnection=True, reconnection_attempts=5)
        self._register_handlers()

    # ── lifecycle ────────────────────────────────────────────────────────
    async def start(self) -> None:
        if self._connected:
            return
        # Server auto-joins the room from handshake auth (same as the viewer).
        await self.sio.connect(
            self.url,
            auth={"peerId": self.peer_id, "roomId": self.room_id},
            transports=["websocket"],
        )
        self._connected = True
        print(f"[rtc] connected to {self.url} room={self.room_id} as {self.peer_id}")

    async def stop(self) -> None:
        for pid in list(self._pcs):
            await self._close_pc(pid)
        self.bus.clear()
        if self._connected:
            self._connected = False
            try:
                await self.sio.disconnect()
            except Exception:
                pass
            print("[rtc] disconnected")

    def frames_flowing(self, peer_id: str, max_age: float = 3.0) -> bool:
        slot = self.bus.get(peer_id)
        return slot is not None and (time.monotonic() - slot[1]) < max_age

    # ── signaling ────────────────────────────────────────────────────────
    def _register_handlers(self) -> None:
        @self.sio.event
        async def connect() -> None:
            print("[rtc] socket connected")

        @self.sio.event
        async def disconnect() -> None:
            print("[rtc] socket disconnected")

        # Handlers take *args: depending on how the server emits, socket.io may
        # deliver extra positional args alongside the payload dict — the payload
        # is always the first dict we find.
        def _payload(args) -> dict:
            for a in args:
                if isinstance(a, dict):
                    return a
            return {}

        @self.sio.on("offer")
        async def on_offer(*args) -> None:
            try:
                await self._handle_offer(_payload(args))
            except Exception as err:
                print(f"[rtc] offer handling failed: {err}")

        @self.sio.on("ice-candidate")
        async def on_ice(*args) -> None:
            try:
                await self._handle_ice(_payload(args))
            except Exception as err:
                print(f"[rtc] ice add failed: {err}")

        @self.sio.on("peer-left")
        async def on_peer_left(*args) -> None:
            pid = _payload(args).get("peerId")
            if pid in self._pcs:
                print(f"[rtc] publisher left: {pid}")
                await self._close_pc(pid)

    async def _handle_offer(self, msg: dict) -> None:
        publisher = msg.get("from")
        desc = msg.get("description") or {}
        if not publisher or not desc.get("sdp"):
            return
        if msg.get("to") not in (None, self.peer_id):
            return  # addressed to another viewer

        print(f"[rtc] offer from {publisher}")
        await self._close_pc(publisher)  # renegotiation replaces the old leg

        ice_servers = [RTCIceServer(urls=u) for u in self.stun_urls]
        if TURN_URLS:
            ice_servers.append(RTCIceServer(
                urls=TURN_URLS, username=TURN_USERNAME, credential=TURN_PASSWORD,
            ))
        pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))
        self._pcs[publisher] = pc

        @pc.on("track")
        def on_track(track) -> None:
            print(f"[rtc] track from {publisher}: {track.kind}")
            # Drain EVERY track (undrained tracks build queues); keep video only.
            asyncio.ensure_future(self._drain(track, publisher))

        @pc.on("connectionstatechange")
        async def on_state() -> None:
            print(f"[rtc] {publisher} connection: {pc.connectionState}")
            if pc.connectionState in ("failed", "closed"):
                await self._close_pc(publisher)

        @pc.on("iceconnectionstatechange")
        async def on_ice_state() -> None:
            print(f"[rtc] {publisher} ice: {pc.iceConnectionState}")

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=desc["sdp"], type=desc["type"])
        )
        answer = await pc.createAnswer()
        # aiortc gathers ICE during setLocalDescription — candidates ship inside
        # the answer SDP (no trickle needed from our side).
        await pc.setLocalDescription(answer)
        await self.sio.emit(
            "answer",
            {
                "roomId": self.room_id,
                "to": publisher,
                "description": {
                    "type": pc.localDescription.type,
                    "sdp": pc.localDescription.sdp,
                },
            },
        )
        print(f"[rtc] answer sent to {publisher}")

    async def _handle_ice(self, msg: dict) -> None:
        publisher = msg.get("from")
        pc = self._pcs.get(publisher)
        cand = (msg.get("candidate") or {})
        cand_str = cand.get("candidate")
        if pc is None or not cand_str:
            return
        # "candidate:foundation 1 udp ..." -> aiortc wants the part after the prefix
        sdp_part = cand_str.split(":", 1)[1] if cand_str.startswith("candidate:") else cand_str
        ice = candidate_from_sdp(sdp_part)
        ice.sdpMid = cand.get("sdpMid")
        ice.sdpMLineIndex = cand.get("sdpMLineIndex")
        await pc.addIceCandidate(ice)

    # ── media ────────────────────────────────────────────────────────────
    async def _drain(self, track, publisher: str) -> None:
        count = 0
        while True:
            try:
                frame = await track.recv()
            except Exception:
                break  # track ended
            if track.kind != "video":
                continue  # drained and dropped (audio etc.)
            self.bus.put(publisher, frame.to_ndarray(format="rgb24"))
            count += 1
            if count == 1:
                print(f"[rtc] first frame from {publisher} "
                      f"({frame.width}x{frame.height})")

    async def _close_pc(self, peer_id: str) -> None:
        pc = self._pcs.pop(peer_id, None)
        if pc is not None:
            try:
                await pc.close()
            except Exception:
                pass
