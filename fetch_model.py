"""Download the lingbot-map checkpoint from HuggingFace if it isn't present.

Weights are 4.6GB — never committed to git. On a RunPod pod with a network
volume, this runs once and the file persists across restarts.

  MODEL_PATH   where to place it (default ./lingbot-map.pt; engine.py reads same)
  MODEL_REPO   HF repo (default robbyant/lingbot-map)
  MODEL_FILE   filename in the repo (default lingbot-map.pt)
"""

import os
import sys
import urllib.request

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "lingbot-map.pt"))
MODEL_REPO = os.environ.get("MODEL_REPO", "robbyant/lingbot-map")
MODEL_FILE = os.environ.get("MODEL_FILE", "lingbot-map.pt")
URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"


def _progress(done: int, total: int) -> None:
    if total > 0:
        pct = done * 100 // total
        mb = done / 1e6
        sys.stdout.write(f"\r  downloading… {pct}% ({mb:.0f} MB)")
        sys.stdout.flush()


def main() -> None:
    if os.path.isfile(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000_000:
        print(f"[fetch_model] already present: {MODEL_PATH} "
              f"({os.path.getsize(MODEL_PATH)/1e9:.1f} GB)")
        return
    print(f"[fetch_model] downloading {URL}\n            → {MODEL_PATH}")
    tmp = MODEL_PATH + ".part"
    with urllib.request.urlopen(URL) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                _progress(done, total)
    os.replace(tmp, MODEL_PATH)
    print(f"\n[fetch_model] done ({os.path.getsize(MODEL_PATH)/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
