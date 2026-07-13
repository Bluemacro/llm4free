"""Live client test for all llm4free.TTI providers.

Runs each provider's images.create() against the real API and verifies the
returned URL actually serves an image. Auth-required providers are skipped
when no credentials are available.
"""
import concurrent.futures as cf
import os
import traceback
from urllib.parse import urlparse

from curl_cffi import requests

import llm4free.TTI as TTI

PROMPT = "a cute cat wearing a hat, studio lighting"

# provider name -> (class, default model, kwargs)
TESTABLE = [
    ("MagicHourAI", TTI.MagicHourAI, "general", {}),
    ("MagicStudioAI", TTI.MagicStudioAI, "magicstudio", {}),
    ("MiragicAI", TTI.MiragicAI, "flux", {}),
    ("NoLoginTool", TTI.NoLoginTool, "@cf/black-forest-labs/flux-1-schnell", {}),
    ("OneFreeAI", TTI.OneFreeAI, "qwen_image_plus", {}),
    ("PerchanceAI", TTI.PerchanceAI, "painted-anime", {}),
    ("PollinationsAI", TTI.PollinationsAI, "flux", {}),
    ("Lexica", TTI.Lexica, "lexica", {}),
    ("StableHordeAI", TTI.StableHordeAI, "auto", {}),
]

# provider name -> reason skipped
AUTH_REQUIRED = {
    "BingImageAI": "requires cookies.json (Bing auth)",
    "RaphaelAI": "requires login cookies / browser harvest",
    "TogetherImage": "requires Together.xyz API key",
    "VisualGPT": "requires cookies.json (VisualGPT auth)",
}

PER_PROVIDER_TIMEOUT = 150


def verify_url(url: str, timeout: int = 30) -> tuple[bool, str]:
    """Confirm a URL serves a real image."""
    try:
        resp = requests.get(url, timeout=timeout, impersonate="chrome", stream=True)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"
        ctype = resp.headers.get("content-type", "")
        if "image/" in ctype or ctype == "application/octet-stream":
            return True, f"OK ({ctype})"
        # Fall back to magic bytes
        head = resp.content[:8]
        if head[:4] == b"\x89PNG" or head[:3] == b"\xff\xd8\xff":
            return True, "OK (magic bytes)"
        return False, f"non-image content-type: {ctype!r}"
    except Exception as e:  # noqa: BLE001
        return False, f"fetch error: {e}"


def run_one(name, cls, model, kwargs):
    try:
        client = cls(**kwargs)
    except FileNotFoundError as e:
        return name, "AUTH/SKIP", str(e).splitlines()[0]
    except Exception as e:  # noqa: BLE001
        return name, "CONSTRUCT-ERROR", f"{type(e).__name__}: {e}"

    try:
        resp = client.images.create(
            model=model,
            prompt=PROMPT,
            n=1,
            response_format="url",
            timeout=PER_PROVIDER_TIMEOUT,
        )
    except Exception as e:  # noqa: BLE001
        return name, "GEN-ERROR", f"{type(e).__name__}: {e}"

    if not getattr(resp, "data", None):
        return name, "EMPTY", "no data returned"

    url = resp.data[0].url
    if not url:
        return name, "NO-URL", "data had empty url"

    ok, detail = verify_url(url)
    if ok:
        return name, "WORKING", f"url verified -> {url[:90]}"
    return name, "BAD-URL", f"{detail} ({url[:90]})"


def main():
    results = {}
    with cf.ThreadPoolExecutor(max_workers=len(TESTABLE)) as ex:
        futs = {
            ex.submit(run_one, n, c, m, k): n for n, c, m, k in TESTABLE
        }
        for fut in cf.as_completed(futs):
            name, status, detail = fut.result()
            results[name] = (status, detail)

    print("=" * 70)
    print("LIVE TTI PROVIDER TEST RESULTS")
    print("=" * 70)
    working, broken = [], []
    for name, (status, detail) in results.items():
        print(f"[{status:14}] {name:16} - {detail}")
        if status == "WORKING":
            working.append(name)
        else:
            broken.append((name, status, detail))

    print("-" * 70)
    print("AUTH-REQUIRED (not tested):")
    for name, reason in AUTH_REQUIRED.items():
        print(f"  SKIP  {name:16} - {reason}")

    print("-" * 70)
    print(f"WORKING ({len(working)}): {', '.join(working) if working else 'none'}")
    print(f"NOT WORKING ({len(broken)}):")
    for name, status, detail in broken:
        print(f"  - {name} [{status}] {detail}")


if __name__ == "__main__":
    main()
