"""
langchain_browserlm/web/navigation.py
=========================
Navigation and general page-readiness actions.
"""

from __future__ import annotations

import asyncio

from playwright.async_api import Page

from langchain_browserlm.core.config import QWEN_URL
import logging

from langchain_browserlm.web.selectors import (
    CHAT_TEXTAREA,
    COOKIE_DIALOG_BUTTONS,
    NEW_CHAT_BUTTON,
)

_log = logging.getLogger("langchain_browserlm.web.nav")


async def dismiss_dialogs(page: Page) -> None:
    """Click away cookie banners and consent overlays."""
    for selector in COOKIE_DIALOG_BUTTONS:
        try:
            btn = page.locator(selector).first
            if await btn.is_visible():
                await btn.click()
                await asyncio.sleep(0.5)
                break
        except Exception:
            pass


async def wait_for_chat_ready(page: Page) -> None:
    await page.wait_for_selector(CHAT_TEXTAREA, timeout=120_000)


async def navigate_home(page: Page, temporary_chat: bool = False) -> None:
    url = f”{QWEN_URL}/?temporary-chat=true” if temporary_chat else QWEN_URL
    try:
        await page.goto(url, wait_until=”domcontentloaded”, timeout=30_000)
    except Exception:
        pass
    await dismiss_dialogs(page)
    await wait_for_chat_ready(page)


async def enable_temporary_chat(page: Page) -> None:
    “””Navigate to the temporary-chat URL if not already active.”””
    if “temporary-chat=true” not in page.url:
        await navigate_home(page, temporary_chat=True)
        _log.info(“Temporary chat enabled via URL.”)


async def click_new_chat(page: Page) -> None:
    await dismiss_dialogs(page)
    btn = page.locator(NEW_CHAT_BUTTON).first
    if await btn.count() > 0:
        await btn.click()
        await asyncio.sleep(0.8)
        await wait_for_chat_ready(page)
    else:
        await navigate_home(page)
