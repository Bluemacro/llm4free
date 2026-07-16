"""
Unfinished / experimental LLM providers.

This subpackage holds providers that are implemented but **not yet reliably
usable** in all environments. They are intentionally excluded from the parent
``llm4free.llm`` package scan and ``__all__``, so the unified client will not
auto-discover or fail over to them.

Why these live here
-------------------
The three providers below drive a real browser (via ``llm4free.requests.cdp``,
backed by the ``agent-browser`` CLI) to solve Cloudflare Turnstile challenges
and harvest cookies / tokens:

  * ``DeepInfraFree``  - solves Turnstile to obtain an ``X-DeepInfra-Turnstile``
                        header for the free DeepInfra API.
  * ``Perchance``      - solves Turnstile to obtain a ``userKey`` for the
                        text-generation.perchance.org API.
  * ``Cloudflare``     - passes the Cloudflare challenge for the AI Playground.

They work structurally (the browser launches, navigates, and evaluates JS
correctly), but Cloudflare's Turnstile does not resolve in headless / sandboxed
environments, so the token / ``userKey`` is never produced. They can be moved
back into ``llm4free.llm`` once a browser environment is available that passes
the challenge (e.g. a headed browser or a fingerprint Cloudflare accepts).

They remain importable directly::

    from llm4free.llm.UNFINISHED import DeepInfraFree, Perchance, Cloudflare
"""

from llm4free.llm.UNFINISHED.cloudflare import Cloudflare
from llm4free.llm.UNFINISHED.deepinfra_free import DeepInfraFree
from llm4free.llm.UNFINISHED.perchance import Perchance

__all__ = [
    "Cloudflare",
    "DeepInfraFree",
    "Perchance",
]
