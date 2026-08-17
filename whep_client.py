"""WHEP subscriber — the pod consumes the phone's stream from MediaMTX.

Replaces rtc_client.RtcSubscriber (socket.io mesh peer) with the SFU that
file's docstring anticipated. Same FrameBus, same interface, so the mapping
loop and server.py's capture path are untouched.

WHY THE SFU, in the two numbers that decided it:

* **Uplink.** In the mesh the phone encoded and uploaded ONE COPY PER PEER.
  Pod plus three viewers at 8Mbps was 32Mbps out of warehouse wifi, which is
  why 2-3 devices was the ceiling. Here the phone publishes once.
* **TURN.** RunPod's NAT is symmetric, so a STUN punch to the phone could
  never succeed and every live frame relayed through metered.ca — a 500MB
  monthly tier, about 16 walks. MediaMTX has a public IP, so the pod dials
  OUTBOUND to it like any other client and TURN disappears entirely. Note the
  absence of RTCConfiguration below: no ICE servers are needed at all.

WHEP is also far less code than the mesh: one POST instead of offer/answer
plus trickle-ICE plus peer bookkeeping. aiortc completes ICE gathering inside
setLocalDescription, so the offer we post is already complete and non-trickle —
which is exactly what MediaMTX's WHEP endpoint expects.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional
from urllib.parse import urljoin

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription

from rtc_client import FrameBus


class WhepSubscriber:
    """Pulls one MediaMTX path into the FrameBus.

    ``peer_id`` doubles as the MediaMTX path: the phone publishes to
    ``/<peer_id>/whip`` and we read ``/<peer_id>/whep``. Peer ids already begin
    with "glasses-", which is what the gateway's regex path matches, so adding
    a device needs no server config.
    """

    def __init__(self, base_url: str, peer_id: str, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.peer_id = peer_id
        self.timeout = timeout
        self.bus = FrameBus()
        self._pc: Optional[RTCPeerConnection] = None
        self._resource: Optional[str] = None
        self._drain_task: Optional[asyncio.Task] = None

    @property
    def whep_url(self) -> str:
        return urljoin(self.base_url, f"{self.peer_id}/whep")

    # ── lifecycle ────────────────────────────────────────────────────────
    async def start(self) -> None:
        # No RTCConfiguration: MediaMTX is on a public IP, so host candidates
        # are directly reachable and neither STUN nor TURN is involved.
        pc = RTCPeerConnection()
        self._pc = pc

        # recvonly — we consume, never publish. Without an explicit transceiver
        # the offer carries no media section and MediaMTX has nothing to answer.
        pc.addTransceiver("video", direction="recvonly")
        # Audio too, even though the mapper only ever reads video frames.
        # MediaMTX matches EVERY published track against the offer and rejects
        # the whole subscription with "codecs not supported by client" if one
        # has no match — so a publisher that happens to carry audio (a browser
        # test page, or a phone build that stops passing audio:false) would
        # otherwise fail in a way that reads like a codec or network problem.
        # The track is received and dropped; on_track only drains video.
        pc.addTransceiver("audio", direction="recvonly")

        @pc.on("track")
        def on_track(track) -> None:
            print(f"[whep] track from {self.peer_id}: {track.kind}")
            if track.kind == "video":
                self._drain_task = asyncio.create_task(self._drain(track))

        @pc.on("connectionstatechange")
        async def on_state() -> None:
            print(f"[whep] {self.peer_id} connection: {pc.connectionState}")

        offer = await pc.createOffer()
        # aiortc finishes ICE gathering inside setLocalDescription, so
        # pc.localDescription is a complete non-trickle offer — which is what
        # WHEP wants. No candidate exchange afterwards.
        await pc.setLocalDescription(offer)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self.whep_url,
                content=pc.localDescription.sdp,
                headers={"Content-Type": "application/sdp"},
            )
        if resp.status_code not in (200, 201):
            await self._close()
            raise RuntimeError(
                f"WHEP POST {self.whep_url} returned {resp.status_code}: "
                f"{resp.text[:200]}"
            )
        # 404 here means nothing is publishing to that path yet — the phone
        # must start before the pod subscribes.
        self._resource = resp.headers.get("location")
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=resp.text, type="answer")
        )
        print(f"[whep] subscribed to {self.whep_url}")

    async def stop(self) -> None:
        # DELETE the resource so MediaMTX tears its side down immediately
        # instead of waiting for ICE to time out.
        if self._resource:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.delete(urljoin(self.whep_url, self._resource))
            except Exception as err:  # noqa: BLE001 - teardown is best-effort
                print(f"[whep] resource delete failed: {err}")
            self._resource = None
        await self._close()
        self.bus.clear()

    async def _close(self) -> None:
        if self._drain_task:
            self._drain_task.cancel()
            self._drain_task = None
        if self._pc is not None:
            try:
                await self._pc.close()
            except Exception:
                pass
            self._pc = None

    # ── media ────────────────────────────────────────────────────────────
    def frames_flowing(self, peer_id: str, max_age: float = 3.0) -> bool:
        slot = self.bus.get(peer_id)
        return slot is not None and (time.monotonic() - slot[1]) < max_age

    async def _drain(self, track) -> None:
        count = 0
        while True:
            try:
                frame = await track.recv()
            except Exception:
                break  # track ended
            self.bus.put(self.peer_id, frame.to_ndarray(format="rgb24"))
            count += 1
            if count == 1:
                # The resolution that actually arrived, not what was requested.
                # Portrait costs the mapper 1369 patches against landscape's
                # 777 — ~6fps versus ~18 — so this line is the check that the
                # phone published landscape.
                print(f"[whep] first frame from {self.peer_id} "
                      f"({frame.width}x{frame.height})")
