"""
langchain_browserlm/runtime/worker.py
=========================
Generic browser worker â€” owns one persistent browser context and serves
queued chat requests by delegating to a BaseBrowserProvider.

The worker knows nothing about Qwen, Kimi, or any specific LLM UI.
All service-specific actions are handled by the injected provider.
"""

from __future__ import annotations

import asyncio

from playwright.async_api import BrowserContext, Page

from langchain_browserlm.core.config import PROFILES_DIR, UserCred
from langchain_browserlm.core.log import worker_logger
from langchain_browserlm.core.requests import QueuedChatRequest
from langchain_browserlm.providers.base import BaseBrowserProvider


class BrowserWorker:
    """Owns one persistent browser context and serves queued chat requests."""

    def __init__(
        self,
        cred: UserCred,
        provider: BaseBrowserProvider,
        headless: bool,
        request_queue: "asyncio.Queue[QueuedChatRequest | None]",
        playwright,
    ) -> None:
        self.cred = cred
        self.worker_id = cred.index
        self.provider = provider
        self.provider.worker_id = cred.index   # inject before any provider call
        self.headless = headless
        self.request_queue = request_queue
        self.playwright = playwright

        self.context: BrowserContext | None = None
        self.ready = asyncio.Event()
        self.startup_error: BaseException | None = None
        self._log = worker_logger(cred.index)

    # â”€â”€ Lifecycle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def run(self) -> None:
        try:
            self.context = await self._launch_context()
            page = await self._get_page()
            self._log.info("Provider setup: %s", self.provider.label)
            await self.provider.setup(page, self.cred)
            self._log.info("Ready. [%s]", self.provider.label)
            self.ready.set()
            await self._serve(page)
        except Exception as exc:
            if not self.ready.is_set():
                self.startup_error = exc
                self.ready.set()
            else:
                self._log.error("Unhandled error: %s", exc, exc_info=True)
        finally:
            await self._close_context()
            self._log.info("Shut down.")

    async def wait_ready(self) -> None:
        """Block until startup succeeds or fails."""
        await self.ready.wait()
        if self.startup_error is not None:
            raise RuntimeError(
                f"Worker {self.worker_id} failed to start."
            ) from self.startup_error

    # â”€â”€ Request loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def _serve(self, page: Page) -> None:
        while True:
            item = await self.request_queue.get()
            try:
                if item is None:
                    return
                await self._handle(page, item)
            finally:
                self.request_queue.task_done()

    async def _handle(self, page: Page, item: QueuedChatRequest) -> None:
        future = item.future
        try:
            self._log.debug("Handling request")
            result = await self.provider.send(page, item.request)
            if not future.done():
                future.set_result(result)
        except Exception as exc:
            self._log.error("Request failed: %s", exc, exc_info=True)
            if not future.done():
                future.set_exception(exc)
            await self.provider.recover(page)

    # â”€â”€ Browser context helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def _launch_context(self) -> BrowserContext:
        profile = PROFILES_DIR / f"worker-{self.worker_id}"
        profile.mkdir(parents=True, exist_ok=True)
        self._log.debug("Launching browser  headless=%s  profile=%s", self.headless, profile)
        return await self.playwright.chromium.launch_persistent_context(
            str(profile),
            headless=self.headless,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
            permissions=["clipboard-read", "clipboard-write"],
        )

    async def _get_page(self) -> Page:
        assert self.context is not None
        return (
            self.context.pages[0]
            if self.context.pages
            else await self.context.new_page()
        )

    async def _close_context(self) -> None:
        if self.context is None:
            return
        try:
            await self.context.close()
        except Exception:
            pass


# Backward-compat alias â€” existing imports of QwenWorker keep working.
QwenWorker = BrowserWorker
