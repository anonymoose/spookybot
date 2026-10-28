"""Long-lived HTTP connections to Anthropic and the speech server.

The Pi's Wi-Fi drops packets (~10% to the router), and a fresh connection pays
for every lost one: a DNS lookup that misses waits out the resolver's 5 s retry,
a lost SYN costs 1 s, and TLS adds round trips on top. Measured from the Pi, a
fresh connection to Anthropic took ~6 s and a kept-open one ~0.3 s. The HTTP
libraries close idle connections after 5 s by default, so nearly every guest
turn paid the 6 s. These clients keep connections for minutes instead, and
SpookyBot pings both services while idle so the connections never go cold.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

KEEPALIVE_S = 300.0
# uvicorn (the speech server) closes connections idle for 5 s, whatever we ask.
SPEECH_PING_S = 4.0


def anthropic_kwargs() -> dict:
    """Extra Anthropic() arguments for a long-lived connection pool."""
    try:
        import anthropic
        import httpx2
        return {"http_client": anthropic.DefaultHttpxClient(
            limits=httpx2.Limits(max_connections=100, max_keepalive_connections=20,
                                 keepalive_expiry=KEEPALIVE_S))}
    except Exception as exc:    # an SDK without httpx2: the default pool still works
        log.debug("default Anthropic HTTP client (%s)", exc)
        return {}


def openai_kwargs() -> dict:
    """Extra OpenAI() arguments (the speech server) for a long-lived pool."""
    try:
        import httpx
        import openai
        return {"http_client": openai.DefaultHttpxClient(
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20,
                                keepalive_expiry=KEEPALIVE_S))}
    except Exception as exc:
        log.debug("default OpenAI HTTP client (%s)", exc)
        return {}
