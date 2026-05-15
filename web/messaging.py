"""Compatibility facade â€” import from langchain_browserlm.web.* submodules directly."""

from langchain_browserlm.web.message_composer import (
    MessageComposer,
    send_message,
    send_message_with_image,
    upload_image,
)
from langchain_browserlm.web.response_reader import (
    ResponseReader,
    extract_response,
    wait_for_response,
)

__all__ = [
    "MessageComposer",
    "ResponseReader",
    "extract_response",
    "send_message",
    "send_message_with_image",
    "upload_image",
    "wait_for_response",
]
