"""
langchain_browserlm/web/selectors.py
========================
CSS/text selectors used by the Qwen web UI automation.
"""

from __future__ import annotations

COOKIE_DIALOG_BUTTONS = (
    "button:has-text('Accept all')",
    "button:has-text('Accept All')",
    "button:has-text('Agree')",
    "button:has-text('Got it')",
    "[class*='cookie'] button",
    "[class*='consent'] button",
    "[class*='banner'] button:has-text('Accept')",
)

CHAT_TEXTAREA = "textarea.message-input-textarea"
NEW_CHAT_BUTTON = "text=New Chat"

LOGIN_BUTTON = ".auth-button-ui.login"
EMAIL_INPUT = 'input[name="email"]'
PASSWORD_INPUT = 'input[name="password"]'
LOGIN_SUBMIT_BUTTON = 'button[type="submit"]'

MODEL_DROPDOWN = "header .ant-dropdown-trigger"
MODEL_SELECTOR_POPUP = ".index-module__model-selector-popup___TGWn8"
MODEL_ITEM_NAME = ".index-module__model-item-name___X8Hec"
MODEL_EXPAND_BUTTON = ".index-module__view-more___iP0nb"

MODEL_REGIME_TRIGGER = (
    "[class*='qwen-select-thinking']:not([class*='dropdown']) "
    ".ant-select-selector"
)
MODEL_REGIME_OPTION = ".ant-select-item-option[title='{regime}']"

UPLOAD_MENU_BUTTON = ".mode-select"
UPLOAD_MENU = ".mode-select-dropdown"
UPLOAD_ATTACHMENT_ITEM = ".mode-select-common-item"

SEND_BUTTON = "button.send-button"
COPY_RESPONSE_BUTTON = ".copy-response-button"

# Temporary / private chat mode â€” disables server-side history storage.
# Located in the page header; present once the user is logged in.
TEMPORARY_CHAT_BUTTON = "div.temporary-chat-entry"
# Class present on the button when temporary-chat mode is already ON.
# Clicking toggles; we check this before clicking to stay idempotent.
TEMPORARY_CHAT_ACTIVE_CLASS = "temporary-chat-entry-out"
