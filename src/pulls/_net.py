"""Network bootstrap for stats.nba.com pulls.

Why this exists
---------------
On this Windows machine, HTTPS to *.nba.com fails certificate verification:

    SSLError: HTTPSConnectionPool(host='stats.nba.com', port=443) ...

Diagnosed 2026-07-20: antivirus/firewall TLS interception. The interceptor presents
its own certificate, which Windows trusts but Python's bundled `certifi` list does not.

The fix is NOT `verify=False` (that disables certificate checking entirely and opens
the door to a real man-in-the-middle). Instead we point Python at the *Windows
certificate store*, which already trusts the interceptor's root CA — so verification
stays ON and succeeds.

Import and call `bootstrap()` once, before any nba_api call:

    from src.pulls._net import bootstrap
    bootstrap()
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

_READY = False

# stats.nba.com silently throttles impatient callers; pace every request.
DEFAULT_SLEEP = 1.0
DEFAULT_TIMEOUT = 45


def bootstrap() -> None:
    """Route SSL verification through the OS trust store. Idempotent."""
    global _READY
    if _READY:
        return
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "truststore is required for stats.nba.com access on this machine "
            "(TLS interception). Install it: pip install truststore"
        ) from exc
    _READY = True


def call(
    endpoint: Callable[..., Any],
    *,
    retries: int = 3,
    sleep: float = DEFAULT_SLEEP,
    timeout: int = DEFAULT_TIMEOUT,
    **kwargs: Any,
):
    """Call an nba_api endpoint with bootstrap, pacing, and retry-on-transient.

    Returns the endpoint's first dataframe, or None if every attempt failed.
    """
    bootstrap()
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            ep = endpoint(timeout=timeout, **kwargs)
            frames = ep.get_data_frames()
            return frames[0] if frames else None
        except Exception as exc:  # noqa: BLE001 - transient network/API errors
            last = exc
            if attempt < retries:
                time.sleep(sleep * attempt * 2)  # simple backoff
        finally:
            time.sleep(sleep)
    print(f"      !! failed after {retries} attempts: {type(last).__name__}: {str(last)[:110]}")
    return None
