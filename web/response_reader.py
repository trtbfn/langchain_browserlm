"""
langchain_browserlm/web/response_reader.py
==============================
Assistant response completion and extraction.
"""

from __future__ import annotations

import asyncio
import json
import re

from playwright.async_api import Page

from langchain_browserlm.core.config import APP_DIR
from langchain_browserlm.core.log import wprint
from langchain_browserlm.web.selectors import COPY_RESPONSE_BUTTON


async def wait_for_response(
    page: Page, worker_id: int, timeout: int = 900_000, baseline: int | None = None
) -> None:
    """Block until the assistant's response is fully generated."""
    await wprint(worker_id, "Waiting for response ...")
    if baseline is None:
        baseline = await page.locator(COPY_RESPONSE_BUTTON).count()
    try:
        await page.wait_for_function(
            f"() => document.querySelectorAll('{COPY_RESPONSE_BUTTON}').length > {baseline}",
            timeout=timeout,
        )
    except Exception:
        shot_path = APP_DIR / f"debug_timeout_w{worker_id}.png"
        try:
            await page.screenshot(path=str(shot_path), full_page=False)
            await wprint(worker_id, f"Timeout screenshot -> {shot_path.name}")
        except Exception:
            pass
        raise
    await wprint(worker_id, "Done.")


def _response_is_complete(text: str) -> bool:
    t = text.strip()

    m = re.match(r"fn\s+(?:\d+\s+)*(.*)", t, re.DOTALL)
    if m:
        body = m.group(1).replace("\xa0", " ").strip()
        try:
            start = body.index("{")
            json.JSONDecoder().raw_decode(body[start:])
            return True
        except (ValueError, IndexError):
            return False

    m_ans = re.match(r"ans\s+(?:\d+\s+)*(.*)", t, re.DOTALL)
    content = m_ans.group(1).strip() if m_ans else t

    if not content or content.isdigit():
        return False

    ends_cleanly = content[-1] in ".?!)" or content.endswith("...")
    return len(content) > 80 or ends_cleanly


async def extract_response(page: Page) -> str:
    """Read the last assistant response from the DOM after streaming stabilizes."""
    js = """
    () => {
        const answers = document.querySelectorAll('.phase-answer');
        if (answers.length) return answers[answers.length - 1].innerText.trim();
        const contents = document.querySelectorAll('.response-message-content');
        if (contents.length) return contents[contents.length - 1].innerText.trim();
        return '';
    }
    """
    await asyncio.sleep(1.5)

    prev = ""
    stable = 0
    for _ in range(354):
        await asyncio.sleep(0.25)
        text: str = await page.evaluate(js)
        if text and text == prev:
            stable += 1
            if stable >= 12 and _response_is_complete(text):
                return text
        else:
            stable = 0
        prev = text
    return prev


class ResponseReader:
    """Wait for and extract the latest assistant response."""

    def __init__(self, page: Page, worker_id: int):
        self.page = page
        self.worker_id = worker_id

    async def current_response_count(self) -> int:
        return await self.page.locator(COPY_RESPONSE_BUTTON).count()

    async def wait_complete(self, timeout: int = 900_000, baseline: int | None = None) -> None:
        await wait_for_response(self.page, self.worker_id, timeout=timeout, baseline=baseline)

    async def extract(self) -> str:
        return await extract_response(self.page)
