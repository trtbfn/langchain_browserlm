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
    TEMPORARY_CHAT_ACTIVE_CLASS,
    TEMPORARY_CHAT_BUTTON,
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


async def navigate_home(page: Page) -> None:
    try:
        await page.goto(QWEN_URL, wait_until="domcontentloaded", timeout=30_000)
    except Exception:
        pass
    await dismiss_dialogs(page)
    await wait_for_chat_ready(page)


async def enable_temporary_chat(page: Page) -> None:
    """Click the temporary-chat toggle if it is not already active.

    Must be called after login and before model selection.  The button is in
    the page header and disables server-side conversation history for this
    session.  Safe to call on every worker startup â€” it is a no-op when the
    mode is already on.
    """
    try:
        btn = page.locator(TEMPORARY_CHAT_BUTTON).first
        await btn.wait_for(state="visible", timeout=20_000)
        # Only click if not already active.
        classes = await btn.get_attribute("class") or ""
        if TEMPORARY_CHAT_ACTIVE_CLASS not in classes:
            await btn.click()
            await asyncio.sleep(0.5)
            _log.info("Temporary chat enabled.")
        else:
            _log.debug("Temporary chat already active â€” skipping click.")
    except Exception as exc:
        _log.warning("Could not enable temporary chat: %s", exc)


async def click_new_chat(page: Page) -> None:
    await dismiss_dialogs(page)
    btn = page.locator(NEW_CHAT_BUTTON).first
    if await btn.count() > 0:
        await btn.click()
        await asyncio.sleep(0.8)
        await wait_for_chat_ready(page)
    else:
        await navigate_home(page)
