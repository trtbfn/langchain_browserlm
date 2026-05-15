"""
langchain_browserlm/web/auth.py
===================
Login flow for the Qwen web UI.
"""

from __future__ import annotations

import asyncio

from playwright.async_api import Page

from langchain_browserlm.core.config import QWEN_URL, UserCred
from langchain_browserlm.core.log import wprint
from langchain_browserlm.web.navigation import wait_for_chat_ready
from langchain_browserlm.web.selectors import (
    EMAIL_INPUT,
    LOGIN_BUTTON,
    LOGIN_SUBMIT_BUTTON,
    PASSWORD_INPUT,
)


async def ensure_logged_in(page: Page, cred: UserCred, worker_id: int) -> None:
    """Auto-login using cred. No-op if the session is already authenticated."""
    await asyncio.sleep(3)

    login_btn = page.locator(LOGIN_BUTTON).first
    try:
        await login_btn.wait_for(state="visible", timeout=8_000)
    except Exception:
        await wprint(worker_id, "Already logged in.")
        return

    await wprint(worker_id, f"Logging in as {cred.login} ...")
    await login_btn.click()
    await asyncio.sleep(1.0)

    email_input = page.locator(EMAIL_INPUT).first
    try:
        await email_input.wait_for(state="visible", timeout=10_000)
    except Exception:
        await page.screenshot(path=f"login_fail_w{worker_id}.png", full_page=True)
        raise
    await email_input.click()
    await email_input.type(str(cred.login), delay=50)
    await asyncio.sleep(0.3)

    password_input = page.locator(PASSWORD_INPUT).first
    await password_input.click()
    await password_input.type(cred.password.get_secret_value(), delay=50)
    await asyncio.sleep(0.5)

    submit_btn = page.locator(LOGIN_SUBMIT_BUTTON).first
    try:
        await submit_btn.wait_for(state="enabled", timeout=5_000)
    except Exception:
        pass
    await submit_btn.click(force=True)

    await wprint(worker_id, "Waiting for login to complete ...")
    try:
        await page.locator(LOGIN_BUTTON).wait_for(state="hidden", timeout=60_000)
    except Exception:
        pass

    await asyncio.sleep(2)
    await page.goto(QWEN_URL, wait_until="domcontentloaded", timeout=30_000)
    await asyncio.sleep(2)
    await wait_for_chat_ready(page)
    await wprint(worker_id, "Login complete.")
