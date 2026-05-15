"""
langchain_browserlm/adapters/langsmith.py
==============================
LangSmith tracing configuration helpers.

LangSmith tracing is automatic for any LangChain component when the right
environment variables are set.  This module provides convenience functions
so you don't have to remember the variable names.

Quick start:
    from langchain_browserlm.adapters.langsmith import configure_langsmith

    configure_langsmith(api_key="ls-...", project="my-qwen-project")

    # Now every langchain_browserlm.adapters.langchain_chat.ChatQwen call is traced.

Manual env vars (alternative to configure_langsmith):
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=ls-...
    LANGCHAIN_PROJECT=my-project         # optional, default "default"
    LANGCHAIN_ENDPOINT=https://...       # optional, default Smith SaaS

Requires:
    pip install langsmith
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator, Optional


def configure_langsmith(
    api_key: str,
    project: str = "qwen-lc",
    endpoint: Optional[str] = None,
    *,
    enabled: bool = True,
) -> None:
    """
    Configure LangSmith tracing by setting environment variables.

    Call once before qwen_start() and all subsequent LangChain calls
    will be traced automatically.

    Args:
        api_key:  Your LangSmith API key (starts with "ls-").
        project:  LangSmith project name.
        endpoint: Custom LangSmith endpoint (omit for cloud SaaS).
        enabled:  Set to False to disable tracing without removing the key.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enabled else "false"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint


def disable_langsmith() -> None:
    """Turn off LangSmith tracing for the current process."""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


def is_tracing_enabled() -> bool:
    """Return True if LangSmith tracing is currently active."""
    return os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"


@contextmanager
def langsmith_trace(
    run_name: str,
    project: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Generator[Any, None, None]:
    """
    Context manager that wraps a block of code in a named LangSmith run.

    Usage:
        with langsmith_trace("my-experiment", tags=["v2"]) as run:
            result = agent.invoke({"messages": [...]})

    Requires langsmith package (pip install langsmith).
    """
    try:
        from langsmith import trace
    except ImportError as exc:
        raise ImportError("langsmith is required: pip install langsmith") from exc

    trace_kwargs: dict[str, Any] = {"name": run_name}
    if project:
        trace_kwargs["project_name"] = project
    if tags:
        trace_kwargs["tags"] = tags
    if metadata:
        trace_kwargs["metadata"] = metadata

    with trace(**trace_kwargs) as run:
        yield run


def get_current_run_id() -> Optional[str]:
    """
    Return the LangSmith run ID of the currently active trace, or None.

    Useful for linking runs together (e.g. logging the run URL).
    """
    try:
        from langsmith.run_helpers import get_current_run_tree
        run = get_current_run_tree()
        return str(run.id) if run else None
    except Exception:
        return None
