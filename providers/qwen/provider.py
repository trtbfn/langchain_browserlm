"""
langchain_browserlm/providers/qwen/provider.py
===================================
Concrete BaseBrowserProvider implementation for chat.qwen.ai.

Delegates all UI actions to the existing web/ modules â€” no logic is
duplicated.  The only new thing here is wiring them together under the
provider interface.
"""

from __future__ import annotations

from playwright.async_api import Page

from langchain_browserlm.core.config import UserCred
from langchain_browserlm.core.log import worker_logger
from langchain_browserlm.core.requests import ChatRequest
from langchain_browserlm.providers.base import BaseBrowserProvider
from langchain_browserlm.web.auth import ensure_logged_in
from langchain_browserlm.web.message_composer import MessageComposer
from langchain_browserlm.web.model_selection import select_model, select_model_regime
from langchain_browserlm.web.navigation import click_new_chat, enable_temporary_chat, navigate_home
from langchain_browserlm.web.response_reader import ResponseReader


class QwenProvider(BaseBrowserProvider):
    """
    Provider for the Qwen web UI (chat.qwen.ai).

    One instance lives inside one browser tab managed by BrowserWorker.
    Model and mode are fixed at construction time; the worker injects
    ``worker_id`` before the first setup() call.

    Args:
        model: Qwen model name, e.g. ``"Qwen3.6-Plus"`` or ``"QwQ-32B"``.
        mode:  Thinking regime â€” ``"Thinking"``, ``"Auto"``, or ``"Fast"``.
    """

    def __init__(self, model: str, mode: str, temporary_chat: bool = True) -> None:
        self._model = model
        self._mode = mode
        self._temporary_chat = temporary_chat

    @property
    def label(self) -> str:
        tag = "+tmp" if self._temporary_chat else ""
        return f"Qwen/{self._model}/{self._mode}{tag}"

    # â”€â”€ Provider interface â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def setup(self, page: Page, cred: UserCred) -> None:
        log = worker_logger(self.worker_id)
        log.info("setup: navigating home")
        await navigate_home(page)
        log.info("setup: logging in as %s", cred.login)
        await ensure_logged_in(page, cred, self.worker_id)
        if self._temporary_chat:
            log.info("setup: enabling temporary chat")
            await enable_temporary_chat(page)
        log.info("setup: selecting model %s", self._model)
        await select_model(page, self._model, self.worker_id)

    async def send(self, page: Page, request: ChatRequest) -> str:
        import time
        log = worker_logger(self.worker_id)

        log.debug("send: starting new chat turn")
        await click_new_chat(page)
        # click_new_chat navigates away, resetting the ?temporary-chat=true URL.
        if self._temporary_chat:
            await enable_temporary_chat(page)
        await select_model(page, self._model, self.worker_id)
        await select_model_regime(page, self._mode, self.worker_id)

        composer = MessageComposer(page)
        reader = ResponseReader(page, self.worker_id)

        baseline = await reader.current_response_count()
        log.info("send: submitting prompt (%d chars)", len(request.text))
        t0 = time.monotonic()
        await composer.send(request)
        await reader.wait_complete(baseline=baseline)
        text = await reader.extract()
        elapsed = time.monotonic() - t0
        log.info("send: response received in %.1fs (%d chars)", elapsed, len(text))
        return text

    async def recover(self, page: Page) -> None:
        try:
            await navigate_home(page)
        except Exception:
            pass

    @classmethod
    def server_command(cls, port, model, mode, env_path, log_level) -> list[str]:
        import sys
        cmd = [
            sys.executable, "-m", "langchain_browserlm.server",
            "--model", model, "--mode", mode,
            "--port", str(port), "--host", "127.0.0.1",
            "--log-level", log_level,
        ]
        if env_path is not None:
            cmd += ["--env", env_path]
        return cmd
