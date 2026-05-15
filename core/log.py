"""
langchain_browserlm/core/log.py
===================
Logging helpers for the langchain_browserlm library.

Logger hierarchy
----------------
    langchain_browserlm                 library root
    langchain_browserlm.pool            BrowserPool / QwenPool lifecycle
    langchain_browserlm.worker.<N>      per-browser-tab worker
    langchain_browserlm.web.auth        login flow
    langchain_browserlm.web.model       model / regime selection
    langchain_browserlm.web.response    response waiting and extraction
    langchain_browserlm.web.nav         page navigation
    langchain_browserlm.chat            ChatQwen adapter (generate / parse)
    langchain_browserlm.server          HTTP server

Enable logs in your script
--------------------------
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    )

Filter to one layer only::

    logging.getLogger("langchain_browserlm.web.response").setLevel(logging.DEBUG)
"""

from __future__ import annotations

import logging
import sys

# Reconfigure stdout/stderr to UTF-8 on Windows so CJK/emoji don't crash
# the console when log messages contain non-ASCII characters.
if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure") and sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# NullHandler on the library root prevents "No handler found" warnings when
# the caller has not configured the logging system.
logging.getLogger("langchain_browserlm").addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return ``langchain_browserlm.<name>`` logger."""
    return logging.getLogger(f"langchain_browserlm.{name}")


def worker_logger(worker_id: int) -> logging.Logger:
    """Return the per-worker logger ``langchain_browserlm.worker.<worker_id>``."""
    return logging.getLogger(f"langchain_browserlm.worker.{worker_id}")


async def wprint(worker_id: int, *args) -> None:
    """Backward-compatible shim â€” routes to the worker logger at INFO level.

    Existing call sites (web/auth.py, web/model_selection.py, etc.) continue
    to work unchanged.  The asyncio.Lock that was here is no longer needed
    because the logging module is already thread-safe.
    """
    msg = " ".join(str(a) for a in args)
    worker_logger(worker_id).info(msg)
