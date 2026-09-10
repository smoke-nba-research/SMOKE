"""Network bootstrap for stats.nba.com pulls.

Why this exists
---------------
On the machine this project was built on, HTTPS to *.nba.com used to fail certificate
verification:

    SSLError: HTTPSConnectionPool(host='stats.nba.com', port=443) ...

Diagnosed 2026-07-20: antivirus/firewall TLS interception on that machine. The
interceptor presented its own certificate, which Windows trusted but Python's bundled
`certifi` list did not.

The fix was NOT `verify=False` (that disables certificate checking entirely and opens
the door to a real man-in-the-middle). Instead we pointed Python at the *Windows
certificate store* via the `truststore` package, which already trusted the
interceptor's root CA — so verification stayed ON and succeeded.

That interception was specific to the old machine and has since been removed there, so
`truststore` is now optional: `bootstrap()` uses it when installed and otherwise falls
back to Python's bundled CA bundle, which is fine on most machines. Install truststore
only if stats.nba.com starts failing certificate verification again because of local
TLS interception.

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
    """Route SSL verification through the OS trust store, if truststore is installed.

    Idempotent. Never raises: truststore is an opt-in fix for local TLS interception,
    not a hard requirement, so a missing install should degrade to Python's bundled CA
    bundle (certifi) with a warning, not block every pull.
    """
    global _READY
    if _READY:
        return
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        print(
            "      !! truststore not installed; using Python's bundled CA bundle for "
            "certificate verification. This is fine on most machines. If stats.nba.com "
            "calls fail with an SSLError, it likely means local TLS interception -- "
            "install truststore (`pip install truststore`) and re-run."
        )
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
