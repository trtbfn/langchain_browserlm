"""
langchain_browserlm/core/requests.py
========================
Runtime request objects used by the worker pool.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4


@dataclass(slots=True, frozen=True)
class ChatRequest:
    """User request data that can be routed to any browser worker."""

    text: str
    image_path: Path | None = None
    video_path: Path | None = None
    request_id: str = field(default_factory=lambda: uuid4().hex[:8])


@dataclass(slots=True)
class QueuedChatRequest:
    """Request envelope owned by the pool queue."""

    request: ChatRequest
    future: asyncio.Future[str]
