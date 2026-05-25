"""
langchain_browserlm/web/message_composer.py
================================
Message entry and attachment upload actions.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from playwright.async_api import Page

from langchain_browserlm.core.requests import ChatRequest
from langchain_browserlm.web.selectors import (
    CHAT_TEXTAREA,
    SEND_BUTTON,
    UPLOAD_ATTACHMENT_ITEM,
    UPLOAD_MENU,
    UPLOAD_MENU_BUTTON,
    UPLOAD_SPINNER,
)


async def upload_image(page: Page, image_path: Path | str) -> None:
    """Upload image_path via the plus dropdown and attachment item."""
    image_path = Path(image_path)
    for attempt in range(3):
        try:
            await page.keyboard.press("Escape")
            await asyncio.sleep(0.2)
        except Exception:
            pass

        await page.locator(UPLOAD_MENU_BUTTON).click()
        await page.locator(UPLOAD_MENU).wait_for(state="visible", timeout=5_000)
        async with page.expect_file_chooser(timeout=10_000) as fc_info:
            await page.locator(UPLOAD_ATTACHMENT_ITEM).first.click()
        fc = await fc_info.value
        await fc.set_files(str(image_path))
        await asyncio.sleep(1.5)

        has_file = await page.evaluate("""
            () => {
                const inputs = document.querySelectorAll('input[type="file"]');
                return Array.from(inputs).some(inp => inp.files && inp.files.length > 0);
            }
        """)
        if has_file:
            return

        thumb = await page.evaluate("""
            () => !!document.querySelector(
                '[class*="image"][class*="preview"], [class*="upload"][class*="preview"], '
                + '[class*="attach"][class*="thumb"], [class*="file-item"], [class*="img-preview"]'
            )
        """)
        if thumb:
            return


async def send_message_with_image(page: Page, text: str, image_path: Path | str) -> None:
    """Upload image_path then send text in the same message."""
    await upload_image(page, image_path)
    await send_message(page, text)


async def upload_video(page: Page, video_path: Path | str) -> None:
    """Upload video_path (or audio-only mp4) via the plus dropdown."""
    video_path = Path(video_path)
    for attempt in range(3):
        try:
            await page.keyboard.press("Escape")
            await asyncio.sleep(0.2)
        except Exception:
            pass

        await page.locator(UPLOAD_MENU_BUTTON).click()
        await page.locator(UPLOAD_MENU).wait_for(state="visible", timeout=5_000)
        async with page.expect_file_chooser(timeout=10_000) as fc_info:
            await page.locator(UPLOAD_ATTACHMENT_ITEM).first.click()
        fc = await fc_info.value
        await fc.set_files(str(video_path))

        # Video files trigger a transcoding spinner; audio-only files attach immediately.
        try:
            await page.wait_for_selector(UPLOAD_SPINNER, timeout=5_000)
            await page.wait_for_selector(UPLOAD_SPINNER, state="hidden", timeout=300_000)
            return
        except Exception:
            pass

        # No spinner — check the DOM for any attached file preview.
        await asyncio.sleep(1.5)
        has_file = await page.evaluate("""
            () => {
                const inputs = document.querySelectorAll('input[type="file"]');
                return Array.from(inputs).some(inp => inp.files && inp.files.length > 0);
            }
        """)
        if has_file:
            return

        file_item = await page.evaluate("""
            () => !!document.querySelector(
                '[class*="file-item"], [class*="video"][class*="preview"], '
                + '[class*="attach"][class*="thumb"], [class*="upload"][class*="preview"]'
            )
        """)
        if file_item:
            return


async def send_message_with_video(page: Page, text: str, video_path: Path | str) -> None:
    """Upload video_path then send text in the same message."""
    await upload_video(page, video_path)
    await send_message(page, text)


async def _textarea_is_empty(page: Page) -> bool:
    try:
        val = await page.locator(CHAT_TEXTAREA).first.input_value()
        return val.strip() == ""
    except Exception:
        return True


async def send_message(page: Page, text: str) -> None:
    textarea = page.locator(CHAT_TEXTAREA).first
    await textarea.click()
    await page.evaluate(
        """(args) => {
            const el = document.querySelector(args.textareaSelector);
            const setter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value'
            ).set;
            setter.call(el, args.value);
            el.dispatchEvent(new Event('input', { bubbles: true }));
        }""",
        {"textareaSelector": CHAT_TEXTAREA, "value": text},
    )
    await asyncio.sleep(0.5)

    send_btn = page.locator(SEND_BUTTON).first

    for attempt in range(4):
        try:
            await send_btn.wait_for(state="enabled", timeout=5_000)
            await send_btn.click()
        except Exception:
            try:
                await textarea.press("Control+Enter")
            except Exception:
                pass

        await asyncio.sleep(1.0)
        if await _textarea_is_empty(page):
            return

        try:
            await page.evaluate(
                "(selector) => document.querySelector(selector)?.click()",
                SEND_BUTTON,
            )
        except Exception:
            pass
        await asyncio.sleep(1.0)
        if await _textarea_is_empty(page):
            return

        await asyncio.sleep(0.5 * (attempt + 1))

    try:
        await textarea.press("Control+Enter")
    except Exception:
        pass


class MessageComposer:
    """High-level message entry and attachment upload actions."""

    def __init__(self, page: Page):
        self.page = page

    async def send(self, request: ChatRequest) -> None:
        if request.video_path is not None:
            await send_message_with_video(self.page, request.text, request.video_path)
        elif request.image_path is not None:
            await send_message_with_image(self.page, request.text, request.image_path)
        else:
            await send_message(self.page, request.text)
