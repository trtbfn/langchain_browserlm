"""
langchain_browserlm/web/page.py
===================
Page-object facade for user-level actions in the Qwen web UI.
"""

from __future__ import annotations

from playwright.async_api import Page

from langchain_browserlm.core.config import UserCred
from langchain_browserlm.core.requests import ChatRequest
from langchain_browserlm.web.auth import ensure_logged_in
from langchain_browserlm.web.message_composer import MessageComposer
from langchain_browserlm.web.model_selection import select_model, select_model_regime
from langchain_browserlm.web.navigation import click_new_chat, navigate_home
from langchain_browserlm.web.response_reader import ResponseReader


class QwenPage:
    """Facade over a Playwright page."""

    def __init__(self, page: Page, worker_id: int):
        self.page = page
        self.worker_id = worker_id
        self.messages = MessageComposer(page)
        self.responses = ResponseReader(page, worker_id)

    async def prepare(self, cred: UserCred, target_model: str) -> None:
        await self.navigate_home()
        await ensure_logged_in(self.page, cred, self.worker_id)
        await select_model(self.page, target_model, self.worker_id)

    async def navigate_home(self) -> None:
        await navigate_home(self.page)

    async def send_request(
        self,
        request: ChatRequest,
        target_model: str,
        target_regime: str,
    ) -> str:
        await click_new_chat(self.page)
        await select_model(self.page, target_model, self.worker_id)
        await select_model_regime(self.page, target_regime, self.worker_id)

        baseline = await self.responses.current_response_count()
        await self.messages.send(request)
        await self.responses.wait_complete(baseline=baseline)
        return await self.responses.extract()
