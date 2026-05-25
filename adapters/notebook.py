"""
langchain_browserlm/adapters/notebook.py
=============================
Synchronous API for Jupyter notebooks and scripts.

Jupyter runs its own asyncio event loop, causing:
  - asyncio.run() to raise "This event loop is already running"
  - Playwright to conflict with the notebook loop

Solution: BrowserPool runs in a background thread with its own dedicated event
loop.  Notebook cells call plain blocking functions â€” no await needed.

Usage (Qwen, backward-compatible)::

    from langchain_browserlm import qwen_start, qwen_send, qwen_stop

    qwen_start()
    reply = qwen_send("Explain backpropagation in simple terms.")
    qwen_stop()

Usage (custom provider)::

    from langchain_browserlm.adapters.notebook import qwen_start, qwen_send, qwen_stop
    from langchain_browserlm.providers.qwen.provider import QwenProvider

    qwen_start(
        provider_factory=lambda: QwenProvider(model="QwQ-32B", mode="Thinking"),
    )
    reply = qwen_send("Think step by step about quantum entanglement.")
    qwen_stop()
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from pathlib import Path
from typing import Callable, Optional

from langchain_browserlm.core.config import (
    ENV_PATH,
    get_headless,
    load_env,
    parse_credentials,
)
from langchain_browserlm.providers.base import BaseBrowserProvider
from langchain_browserlm.runtime.pool import BrowserPool, QwenPool

_log = logging.getLogger("langchain_browserlm.notebook")

# â”€â”€ Background event loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_bg_loop:   Optional[asyncio.AbstractEventLoop] = None
_bg_thread: Optional[threading.Thread]          = None
_pool:      Optional[BrowserPool]               = None
_bg_lock    = threading.Lock()
_start_lock = threading.Lock()


def _ensure_bg_loop() -> asyncio.AbstractEventLoop:
    """Start the background thread + event loop if not already running."""
    global _bg_loop, _bg_thread
    with _bg_lock:
        if _bg_loop is not None and _bg_loop.is_running():
            return _bg_loop

        loop = asyncio.ProactorEventLoop()  # required on Windows for subprocess support
        _bg_loop = loop
        ready = threading.Event()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.call_soon(ready.set)
            loop.run_forever()

        _bg_thread = threading.Thread(target=_run, name="qwen-bg-loop", daemon=True)
        _bg_thread.start()
        ready.wait(timeout=10)
        return loop


def _submit(coro) -> concurrent.futures.Future:
    """Schedule a coroutine on the background loop."""
    return asyncio.run_coroutine_threadsafe(coro, _ensure_bg_loop())


# â”€â”€ Public sync functions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def qwen_start(
    env_path: Optional[Path] = None,
    provider_factory: Optional[Callable[[], BaseBrowserProvider]] = None,
    timeout: float = 180.0,
) -> None:
    """Open browser windows, log in, and select the model.

    Blocks until every window is ready.  Safe to call again â€” no-op if already
    running.

    Args:
        env_path:         Path to a .env file with credentials (and MODEL /
                          MODEL_MODE when no provider_factory is given).
        provider_factory: Optional callable returning a BaseBrowserProvider.
                          When omitted, a QwenProvider is created from the
                          .env MODEL / MODEL_MODE values (backward compat).
        timeout:          Seconds to wait for all workers to become ready.
    """
    global _pool
    with _start_lock:
        if _pool is not None:
            _log.info("Pool already running â€” qwen_start() is a no-op.")
            return

        async def _start() -> None:
            global _pool
            if provider_factory is not None:
                env = load_env(env_path or ENV_PATH)
                creds = parse_credentials(env)
                headless = get_headless(env)
                if not creds:
                    raise ValueError(
                        f"No credentials found in .env: {env_path or ENV_PATH}"
                    )
                p = BrowserPool(
                    provider_factory=provider_factory,
                    creds=creds,
                    headless=headless,
                )
            else:
                p = QwenPool(env_path=env_path)
            await p.__aenter__()
            _pool = p

        _submit(_start()).result(timeout=timeout)

    _log.info("Pool ready: %d window(s)", _pool.n_workers)


def qwen_send(prompt: str, timeout: float = 900.0, request_id: str | None = None) -> str:
    """Send *prompt* to a free browser window and return the model's reply.

    Blocks until the full response has been received.

    Args:
        prompt:     The text prompt to send.
        timeout:    Seconds to wait for the response.
        request_id: Optional identifier to tag this request for tracking.
                    Auto-generated if not provided.
    """
    if _pool is None:
        raise RuntimeError("Call qwen_start() first to open the browser windows.")
    _log.debug("qwen_send: %d chars  id=%s", len(prompt), request_id)
    return _submit(_pool.send(prompt, request_id=request_id)).result(timeout=timeout)


def qwen_send_with_image(
    prompt: str, image_path: str | Path, timeout: float = 900.0
) -> str:
    """Send *prompt* with an image attachment and return the model's reply."""
    if _pool is None:
        raise RuntimeError("Call qwen_start() first to open the browser windows.")
    return _submit(_pool.send_with_image(prompt, image_path)).result(timeout=timeout)


def qwen_send_with_video(
    prompt: str, video_path: str | Path, timeout: float = 900.0, request_id: str | None = None
) -> str:
    """Upload *video_path*, send *prompt*, and return the model's reply.

    Blocks until the full response has been received.  The video is uploaded
    via the browser file-chooser and the upload spinner is awaited before the
    prompt is submitted.

    Args:
        prompt:     The text prompt to send alongside the video.
        video_path: Local path to the .mp4 (or other video) file.
        timeout:    Seconds to wait for the response (default 15 min).
        request_id: Optional identifier for tracking.
    """
    if _pool is None:
        raise RuntimeError("Call qwen_start() first to open the browser windows.")
    return _submit(
        _pool.send_with_video(prompt, video_path, request_id=request_id)
    ).result(timeout=timeout)


def qwen_stop(timeout: float = 30.0) -> None:
    """Close all browser windows and shut down the pool."""
    global _pool, _bg_loop
    if _pool is None:
        return

    async def _stop() -> None:
        global _pool
        await _pool.__aexit__(None, None, None)
        _pool = None

    _submit(_stop()).result(timeout=timeout)

    if _bg_loop is not None:
        _bg_loop.call_soon_threadsafe(_bg_loop.stop)
        _bg_loop = None

    _log.info("Pool stopped.")
