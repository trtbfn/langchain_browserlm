"""
langchain_browserlm/adapters/langchain_server.py
=================================================
ChatLLM -- a ChatOpenAI subclass that auto-starts a browser-automation server.

    from langchain_browserlm import ChatLLM
    from langchain_browserlm.providers.qwen.provider import QwenProvider

    llm_worker     = ChatLLM(QwenProvider, model="Qwen3.6-Plus", mode="Fast",     env_path=".env")
    llm_supervisor = ChatLLM(QwenProvider, model="QwQ-32B",      mode="Thinking", env_path=".env")

Each instance starts one subprocess on a free local port and returns a fully
standard ChatOpenAI -- bind_tools(), with_structured_output(), LangSmith
tracing, and streaming all work without modification.  All subprocesses are
terminated automatically on process exit.
"""

from __future__ import annotations

import atexit
import logging
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional, Type

from langchain_openai import ChatOpenAI

from langchain_browserlm.providers.base import BaseBrowserProvider

_log = logging.getLogger("langchain_browserlm.chat")
_servers: list[subprocess.Popen] = []


@atexit.register
def _shutdown_all() -> None:
    for proc in _servers:
        try:
            proc.terminate()
        except Exception:
            pass


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(3.0)
    raise TimeoutError(f"Server on port {port} did not respond within {timeout}s.")


class ChatLLM(ChatOpenAI):
    """
    ChatOpenAI backed by any browser-automation server.

        from langchain_browserlm import ChatLLM
        from langchain_browserlm.providers.qwen.provider import QwenProvider

        llm_worker     = ChatLLM(QwenProvider, model="Qwen3.6-Plus", mode="Fast",     env_path=".env")
        llm_supervisor = ChatLLM(QwenProvider, model="QwQ-32B",      mode="Thinking", env_path=".env")
    """

    def __init__(
        self,
        provider: Type[BaseBrowserProvider],
        model: str = "Qwen3.6-Plus",
        mode: str = "Auto",
        env_path: Optional[str | Path] = None,
        startup_timeout: float = 300.0,
        log_level: str = "INFO",
        temporary_chat: bool = True,
        **kwargs: Any,
    ) -> None:
        port = _free_port()
        cmd = provider.server_command(
            port=port,
            model=model,
            mode=mode,
            env_path=str(env_path) if env_path else None,
            log_level=log_level,
            temporary_chat=temporary_chat,
        )
        _log.info("ChatLLM  provider=%s  model=%s  port=%d", provider.__name__, model, port)
        _servers.append(subprocess.Popen(cmd))
        _wait_ready(port, startup_timeout)
        _log.info("Ready on port %d", port)

        super().__init__(model=model, base_url=f"http://127.0.0.1:{port}/v1", api_key="none", **kwargs)
