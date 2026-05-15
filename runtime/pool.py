"""
langchain_browserlm/runtime/pool.py
=======================
BrowserPool â€” generic async pool that works with any BaseBrowserProvider.
QwenPool    â€” convenience subclass that reads config from .env (backward compat).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional

from playwright.async_api import async_playwright

from langchain_browserlm.core.config import (
    ENV_PATH,
    get_headless,
    get_model,
    get_model_regime,
    load_env,
    parse_credentials,
    UserCred,
)
from langchain_browserlm.core.requests import ChatRequest, QueuedChatRequest
from langchain_browserlm.providers.base import BaseBrowserProvider
from langchain_browserlm.runtime.worker import BrowserWorker

_log = logging.getLogger("langchain_browserlm.pool")


class BrowserPool:
    """
    Async context manager that manages N browser workers, one per credential.

    Each worker runs one BaseBrowserProvider instance inside a dedicated
    persistent browser context.  Requests are load-balanced across workers
    via a shared asyncio queue.

    Usage::

        from langchain_browserlm.providers.qwen.provider import QwenProvider

        async with BrowserPool(
            provider_factory=lambda: QwenProvider(model="QwQ-32B", mode="Thinking"),
            creds=[cred1, cred2],
            headless=True,
        ) as pool:
            reply = await pool.send("Explain backpropagation.")
    """

    def __init__(
        self,
        provider_factory: Callable[[], BaseBrowserProvider],
        creds: list[UserCred],
        headless: bool = False,
    ) -> None:
        if not creds:
            raise ValueError("BrowserPool requires at least one UserCred.")
        self._provider_factory = provider_factory
        self._creds = creds
        self._headless = headless
        self._request_queue: asyncio.Queue[QueuedChatRequest | None] = asyncio.Queue()
        self._workers: list[BrowserWorker] = []
        self._worker_tasks: list[asyncio.Task] = []
        self._pw_ctx = None

    async def __aenter__(self) -> "BrowserPool":
        self._pw_ctx = async_playwright()
        pw = await self._pw_ctx.__aenter__()

        _log.info("Starting %d worker(s)", len(self._creds))
        for cred in self._creds:
            provider = self._provider_factory()
            worker = BrowserWorker(
                cred=cred,
                provider=provider,
                headless=self._headless,
                request_queue=self._request_queue,
                playwright=pw,
            )
            self._workers.append(worker)
            self._worker_tasks.append(asyncio.create_task(worker.run()))

        try:
            await asyncio.gather(*(w.wait_ready() for w in self._workers))
        except Exception:
            await self.__aexit__(None, None, None)
            raise

        _log.info("All %d worker(s) ready.", len(self._workers))
        return self

    async def __aexit__(self, *args) -> None:
        for _ in self._workers:
            await self._request_queue.put(None)
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        if self._pw_ctx:
            await self._pw_ctx.__aexit__(*args)
        _log.info("Pool stopped.")

    async def send(self, text: str) -> str:
        """Route text to the next free worker and return the reply."""
        return await self.send_with_image(text, None)

    async def send_with_image(self, text: str, image_path: Optional[Path | str]) -> str:
        """Upload image_path, send text, and return the reply."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        resolved = Path(image_path) if image_path is not None else None
        request = ChatRequest(text=text, image_path=resolved)
        _log.debug("Request queued (%d chars)", len(text))
        await self._request_queue.put(QueuedChatRequest(request=request, future=future))
        return await future

    @property
    def n_workers(self) -> int:
        return len(self._workers)


class QwenPool(BrowserPool):
    """
    Backward-compatible pool that reads MODEL / MODEL_MODE / credentials from
    a .env file and creates a QwenProvider automatically.

    Existing code using ``QwenPool(env_path=...)`` continues to work unchanged.
    """

    def __init__(self, env_path: Optional[Path] = None) -> None:
        from langchain_browserlm.providers.qwen.provider import QwenProvider

        env = load_env(env_path or ENV_PATH)
        model = get_model(env)
        mode = get_model_regime(env)
        headless = get_headless(env)
        creds = parse_credentials(env)

        if not creds:
            raise ValueError(
                "No USER{n}_LOGIN / USER{n}_PASSWORD pairs found in .env\n"
                f"  Looked at: {env_path or ENV_PATH}"
            )

        super().__init__(
            provider_factory=lambda: QwenProvider(model=model, mode=mode),
            creds=creds,
            headless=headless,
        )
        self._model = model
        self._mode = mode

    # Keep the old named properties that callers depend on.
    @property
    def model(self) -> str:
        return self._model

    @property
    def regime(self) -> str:
        return self._mode
