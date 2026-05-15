"""
langchain_browserlm/web/model_selection.py
==============================
Model and mode selection actions for the Qwen web UI.
"""

from __future__ import annotations

import asyncio

from playwright.async_api import Page

from langchain_browserlm.core.config import _VALID_REGIMES
from langchain_browserlm.core.log import wprint
from langchain_browserlm.web.selectors import (
    MODEL_DROPDOWN,
    MODEL_EXPAND_BUTTON,
    MODEL_ITEM_NAME,
    MODEL_REGIME_OPTION,
    MODEL_REGIME_TRIGGER,
    MODEL_SELECTOR_POPUP,
)


async def select_model(page: Page, target_model: str, worker_id: int) -> None:
    """Choose target_model from the model dropdown."""
    model_btn = page.locator(MODEL_DROPDOWN).first
    try:
        current = (await model_btn.inner_text()).strip()
        if target_model in current:
            return
    except Exception:
        pass

    for _attempt in range(2):
        try:
            await model_btn.click()
        except Exception:
            break
        await asyncio.sleep(1.0)

        popup = page.locator(MODEL_SELECTOR_POPUP).first
        try:
            await popup.wait_for(state="visible", timeout=5_000)
        except Exception:
            await page.keyboard.press("Escape")
            await asyncio.sleep(0.5)
            continue

        name_els = popup.locator(MODEL_ITEM_NAME)
        found = False
        for i in range(await name_els.count()):
            el = name_els.nth(i)
            if target_model in (await el.inner_text()).strip():
                await el.click()
                found = True
                await asyncio.sleep(0.5)
                break

        if found:
            return

        expand_btn = popup.locator(MODEL_EXPAND_BUTTON).first
        if await expand_btn.count() > 0 and await expand_btn.is_visible():
            await expand_btn.click()
            await asyncio.sleep(1.0)
            item = page.get_by_text(target_model, exact=True).first
            try:
                await item.wait_for(state="visible", timeout=5_000)
                await item.click()
                await asyncio.sleep(0.5)
                return
            except Exception:
                pass

        await page.keyboard.press("Escape")
        await asyncio.sleep(0.5)

    await wprint(worker_id, f"WARNING: '{target_model}' not found - proceeding with default.")


async def select_model_regime(page: Page, regime: str, worker_id: int) -> None:
    """Set the input-bar thinking mode to one of: Thinking, Auto, Fast."""
    if regime not in _VALID_REGIMES:
        await wprint(worker_id, f"WARNING: unknown regime {regime!r} - skipping.")
        return
    try:
        trigger = page.locator(MODEL_REGIME_TRIGGER).first
        await trigger.wait_for(state="visible", timeout=5_000)
        current = (await trigger.inner_text()).strip()
        if regime.lower() in current.lower():
            return
        await trigger.click()
        await asyncio.sleep(0.5)
        option = page.locator(MODEL_REGIME_OPTION.format(regime=regime)).first
        await option.wait_for(state="visible", timeout=5_000)
        await option.click()
        await asyncio.sleep(0.3)
        await wprint(worker_id, f"Regime set to {regime}.")
    except Exception as exc:
        await wprint(worker_id, f"WARNING: could not set regime '{regime}': {exc}")
