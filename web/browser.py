"""Compatibility facade â€” import from langchain_browserlm.web.* submodules directly."""

from langchain_browserlm.web.auth import ensure_logged_in
from langchain_browserlm.web.model_selection import select_model, select_model_regime
from langchain_browserlm.web.navigation import (
    click_new_chat,
    dismiss_dialogs,
    navigate_home,
    wait_for_chat_ready,
)

__all__ = [
    "click_new_chat",
    "dismiss_dialogs",
    "ensure_logged_in",
    "navigate_home",
    "select_model",
    "select_model_regime",
    "wait_for_chat_ready",
]
