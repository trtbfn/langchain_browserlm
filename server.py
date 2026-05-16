"""
langchain_browserlm/server.py
==============================
OpenAI-compatible FastAPI server wrapping a BrowserPool.

ChatLLM starts this automatically — you only need to run it manually if you
want to point your own ChatOpenAI instance at it directly.

Manual usage (one process per model tier):

    python -m langchain_browserlm.server \
        --model Qwen3.6-Plus --mode Fast --port 8001 --env path/to/.env

    python -m langchain_browserlm.server \
        --model QwQ-32B --mode Thinking --port 8002 --env path/to/.env

    from langchain_openai import ChatOpenAI
    llm_worker     = ChatOpenAI(base_url="http://localhost:8001/v1", api_key="none")
    llm_supervisor = ChatOpenAI(base_url="http://localhost:8002/v1", api_key="none")

Endpoints
---------
    POST /v1/chat/completions   chat with optional tool calling
    GET  /v1/models             returns the configured model name

Tool calling
------------
Translates OpenAI tool schemas into the fn/ans prompt protocol the provider
understands, then maps the response back to OpenAI tool_calls format.
Use ChatOpenAI(...).bind_tools([...]) as normal.

Logging
-------
Pass --log-level DEBUG to see full prompt/response text.
Default is INFO (request in/out with timing and char counts).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

_log = logging.getLogger("langchain_browserlm.server")

# â”€â”€ Pydantic schemas for the OpenAI wire format â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class _Message(BaseModel):
    role: str
    content: Optional[str] = ""
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list] = None
    name: Optional[str] = None


class _ToolFunction(BaseModel):
    name: str
    description: str = ""
    parameters: dict = {}


class _Tool(BaseModel):
    type: str = "function"
    function: _ToolFunction


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[_Message]
    tools: Optional[list[_Tool]] = None
    tool_choice: Optional[Any] = None
    stream: bool = False
    # Silently ignore OpenAI-specific params that don't apply here.
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# â”€â”€ Prompt construction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_TOOL_SYSTEM_BLOCK = """\

You have access to the following Python-style functions:

{schemas}

HOW TO USE THEM:
- To call a function, wrap it in a fenced code block tagged "fn" and nothing else:
  ```fn
  {{"name":"<function_name>","args":{{...}}}}
  ```
- To give a final text answer (no function call), wrap it in a fenced code block tagged "ans":
  ```ans
  your answer here
  ```
- Never output bare JSON outside of a code block.
- Output exactly ONE fenced block per response.
"""


def _tool_schema_line(tool: _Tool) -> str:
    props = tool.function.parameters.get("properties", {})
    args_str = ", ".join(
        f"{k}: {v.get('type', 'any')}" for k, v in props.items()
    )
    return f"- {tool.function.name}({args_str}): {tool.function.description}"


def _build_prompt(messages: list[_Message], tools: list[_Tool] | None) -> str:
    """Convert OpenAI-format messages + tool schemas to the flat prompt Qwen expects."""
    tool_block = (
        _TOOL_SYSTEM_BLOCK.format(
            schemas="\n".join(_tool_schema_line(t) for t in tools)
        )
        if tools
        else ""
    )

    parts: list[str] = []
    has_system = any(m.role == "system" for m in messages)

    if tool_block and not has_system:
        parts.append(tool_block.strip())

    for msg in messages:
        if msg.role == "system":
            parts.append(f"{msg.content}{tool_block}")

        elif msg.role == "user":
            parts.append(f"USER: {msg.content}")

        elif msg.role == "assistant":
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                fn = tc.get("function", {})
                fn_args = fn.get("arguments", "{}")
                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except json.JSONDecodeError:
                        fn_args = {}
                obj = {"name": fn.get("name", ""), "args": fn_args}
                parts.append(f"ASSISTANT:\n```fn\n{json.dumps(obj)}\n```")
            else:
                parts.append(f"ASSISTANT:\n```ans\n{msg.content or ''}\n```")

        elif msg.role == "tool":
            parts.append(f"TOOL_RESULT[{msg.tool_call_id}]: {msg.content}")

    return "\n\n".join(parts)


# â”€â”€ Response parsing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _parse_raw(text: str) -> dict:
    """Parse Qwen's raw response into an OpenAI choice dict."""
    from langchain_browserlm.adapters.langchain_chat import _parse_response
    ai_msg = _parse_response(text)

    if ai_msg.tool_calls:
        tc = ai_msg.tool_calls[0]
        return {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }

    return {
        "message": {
            "role": "assistant",
            "content": ai_msg.content,
        },
        "finish_reason": "stop",
    }


# â”€â”€ Server state (set by main before uvicorn.run) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_pool = None          # BrowserPool instance, live during lifespan
_server_model = "unknown"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Pool is already started before uvicorn; just yield and stop on shutdown.
    _log.info("Server ready. model=%s", _server_model)
    yield
    if _pool is not None:
        await _pool.__aexit__(None, None, None)
    _log.info("Server stopped.")


app = FastAPI(title="langchain_browserlm OpenAI-compatible server", lifespan=_lifespan)


# â”€â”€ Endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": _server_model, "object": "model", "owned_by": "langchain_browserlm"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if _pool is None:
        raise HTTPException(status_code=503, detail="Pool not initialised.")
    if req.stream:
        raise HTTPException(status_code=501, detail="Streaming is not supported.")

    req_id = uuid.uuid4().hex[:10]
    n_tools = len(req.tools) if req.tools else 0
    _log.info("[%s] â†’ %d messages, %d tool(s)", req_id, len(req.messages), n_tools)

    prompt = _build_prompt(req.messages, req.tools)
    _log.debug("[%s] prompt (%d chars): %.300s", req_id, len(prompt), prompt)

    t0 = time.monotonic()
    try:
        raw = await _pool.send(prompt)
    except Exception as exc:
        _log.error("[%s] pool error: %s", req_id, exc, exc_info=True)
        raise HTTPException(status_code=503, detail=str(exc))

    elapsed = time.monotonic() - t0
    _log.info("[%s] â† %d chars in %.1fs", req_id, len(raw), elapsed)
    _log.debug("[%s] raw: %.400s", req_id, raw)

    choice = _parse_raw(raw)
    _log.info("[%s] parsed â†’ %s", req_id, choice["finish_reason"])

    return JSONResponse({
        "id": f"chatcmpl-{req_id}",
        "object": "chat.completion",
        "model": _server_model,
        "choices": [{"index": 0, **choice}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


# â”€â”€ CLI entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="langchain_browserlm OpenAI-compatible server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--provider", default="qwen",          help="Provider name (currently: qwen)")
    p.add_argument("--model",    default="Qwen3.6-Plus",  help="Model name passed to the provider")
    p.add_argument("--mode",     default="Auto",          help="Qwen mode: Thinking | Auto | Fast")
    p.add_argument("--port",     type=int, default=8000,  help="HTTP port")
    p.add_argument("--host",     default="127.0.0.1",     help="Host to bind")
    p.add_argument("--env",      default=None,            help="Path to .env file with credentials")
    p.add_argument("--log-level", default="INFO",         help="DEBUG | INFO | WARNING | ERROR")
    p.add_argument("--no-temporary-chat", action="store_true", default=False,
                   help="Disable temporary chat mode (conversations will be saved)")
    return p.parse_args()


if __name__ == "__main__":
    import asyncio

    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    )

    # â”€â”€ Resolve provider factory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if args.provider == "qwen":
        from langchain_browserlm.providers.qwen.provider import QwenProvider
        provider_factory = lambda: QwenProvider(model=args.model, mode=args.mode, temporary_chat=not args.no_temporary_chat)
    else:
        raise SystemExit(f"Unknown provider: {args.provider!r}. Supported: qwen")

    # â”€â”€ Load credentials from .env â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    from langchain_browserlm.core.config import load_env, parse_credentials, get_headless, ENV_PATH
    env_path = Path(args.env) if args.env else None
    env = load_env(env_path or ENV_PATH)
    creds = parse_credentials(env)
    headless = get_headless(env)

    if not creds:
        raise SystemExit(f"No credentials found in .env: {env_path or ENV_PATH}")

    # â”€â”€ Start pool, then uvicorn â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    from langchain_browserlm.runtime.pool import BrowserPool

    _server_model = args.model

    async def _main():
        global _pool
        async with BrowserPool(
            provider_factory=provider_factory,
            creds=creds,
            headless=headless,
        ) as pool:
            _pool = pool
            config = uvicorn.Config(
                app,
                host=args.host,
                port=args.port,
                log_level=args.log_level.lower(),
                lifespan="on",
            )
            server = uvicorn.Server(config)
            _log.info("Listening on %s:%d  model=%s  mode=%s",
                      args.host, args.port, args.model, args.mode)
            await server.serve()

    asyncio.run(_main())
