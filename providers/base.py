"""
langchain_browserlm/providers/base.py
=========================
Abstract interface every browser-LLM provider must implement.

A provider owns the UI-specific logic for one service (Qwen, Kimi, â€¦).
The generic runtime (BrowserPool / BrowserWorker) handles browser lifecycle,
queuing, and error recovery â€” it delegates all LLM-specific actions to the
provider via this interface.

Implementing a new provider
---------------------------
1. Subclass BaseBrowserProvider.
2. Implement setup(), send(), and recover().
3. Pass a factory to BrowserPool:

    pool = BrowserPool(
        provider_factory=lambda: MyProvider(model="..."),
        creds=[...],
        headless=True,
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Page
    from langchain_browserlm.core.config import UserCred
    from langchain_browserlm.core.requests import ChatRequest


class BaseBrowserProvider(ABC):
    """
    Encapsulates all UI-specific interactions for one browser-hosted LLM.

    One provider instance is created per browser tab (worker).  The runtime
    injects ``worker_id`` before calling setup(), so providers can use it for
    logging without accepting it as a method parameter.
    """

    #: Injected by the worker before setup() is called.
    worker_id: int = 0

    @abstractmethod
    async def setup(self, page: "Page", cred: "UserCred") -> None:
        """Navigate to the service, log in, and select the model.

        Called once when the worker starts.  Must leave the page in a state
        where send() can be called immediately.
        """

    @abstractmethod
    async def send(self, page: "Page", request: "ChatRequest") -> str:
        """Submit *request* and return the complete response text.

        The provider is responsible for starting a fresh conversation turn
        (e.g. clicking "New chat"), submitting the message, waiting for the
        response to finish streaming, and returning the raw text.
        """

    @abstractmethod
    async def recover(self, page: "Page") -> None:
        """Return the page to a clean, ready state after a failed request.

        Must not raise â€” if recovery itself fails, swallow the exception so
        the worker loop can continue serving subsequent requests.
        """

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable identifier for logs, e.g. ``'Qwen/QwQ-32B/Thinking'``."""

    @classmethod
    def server_command(
        cls,
        port: int,
        model: str,
        mode: str,
        env_path: "str | None",
        log_level: str,
    ) -> list[str]:
        """Return the command list to launch this provider's OpenAI-compatible server.

        Override in each concrete provider.  Used by ChatLLM to start the
        background server process.
        """
        raise NotImplementedError(f"{cls.__name__} must implement server_command()")
