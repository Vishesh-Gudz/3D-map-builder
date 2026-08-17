"""Prove the WHEP client works, without the pod, the GPU or EC2.

Run MediaMTX anywhere reachable (your own laptop is fine), publish to it from a
browser, then point this at the same path. It reports the resolution and frame
rate actually received — which is all whep_client.py has to get right.

    # 1. MediaMTX (single binary, no Docker needed)
    ./mediamtx                       # serves :8889

    # 2. browser: publish your webcam
    #    http://localhost:8889/glasses-test/publish

    # 3. this script
    python check_whep.py http://localhost:8889 glasses-test

Needs only aiortc + httpx — no torch, no checkpoint, no CUDA. So it runs on a
laptop, and validating it there means the pod only has to be right about
networking, not about the protocol.
"""

from __future__ import annotations

import asyncio
import sys
import time

from whep_client import WhepSubscriber


async def main(base_url: str, path: str, seconds: float) -> int:
    sub = WhepSubscriber(base_url, path)
    print(f"subscribing to {sub.whep_url}")
    try:
        await sub.start()
    except Exception as err:
        print(f"FAIL: {err}")
        print("\nmost likely causes, in order:")
        print("  404  nothing is publishing to that path — start the browser first")
        print("  connection refused  MediaMTX not running, or wrong host/port")
        print("  timeout  TCP 8889 blocked")
        return 1

    # Signalling succeeding tells you almost nothing — the failure mode that
    # matters is a connection that negotiates fine and then never delivers
    # media, which is what a blocked UDP media port looks like.
    t0 = time.monotonic()
    last_seen = 0.0
    frames = 0
    while time.monotonic() - t0 < seconds:
        slot = sub.bus.get(path)
        if slot is not None and slot[1] > last_seen:
            last_seen = slot[1]
            frames += 1
        await asyncio.sleep(0.005)
    await sub.stop()

    fps = frames / seconds
    print(f"\nreceived {frames} frames in {seconds:.0f}s = {fps:.1f} fps")
    if frames == 0:
        print("FAIL: signalling worked but NO MEDIA arrived.")
        print("  → UDP 8189 is not reachable from here. On EC2 that is the")
        print("    security group; locally it is usually a firewall prompt.")
        print("  → or MTX_WEBRTCADDITIONALHOSTS is unset, so MediaMTX")
        print("    advertised a private IP as its ICE candidate.")
        return 1
    print("PASS — whep_client.py negotiates and receives media.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    sys.exit(asyncio.run(main(sys.argv[1], sys.argv[2], dur)))
