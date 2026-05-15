"""
langchain_browserlm/adapters/mcp/tools.py
===============================
Load tools from an MCP server and expose them as LangChain BaseTool instances,
so ChatQwen (or any LangChain model) can call them via bind_tools().

Usage â€” connect to a local MCP server over stdio:
    import asyncio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_browserlm.adapters.mcp.tools import load_mcp_tools

    async def main():
        params = StdioServerParameters(command="python", args=["-m", "my_mcp_server"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)

        from langchain_browserlm import qwen_start, qwen_stop
        from langchain_browserlm.adapters.langchain_chat import ChatQwen

        qwen_start()
        llm = ChatQwen().bind_tools(tools)
        result = llm.invoke([HumanMessage(content="call my tool")])
        qwen_stop()

    asyncio.run(main())

Usage â€” convenience wrapper for stdio servers:
    from langchain_browserlm.adapters.mcp.tools import mcp_tools_from_stdio

    tools = asyncio.run(
        mcp_tools_from_stdio(command="python", args=["-m", "my_mcp_server"])
    )

Requires:
    pip install mcp
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, create_model


# â”€â”€ Schema â†’ Pydantic model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_JSON_TO_PYTHON: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _json_schema_to_pydantic(name: str, schema: dict[str, Any]) -> Type[BaseModel]:
    """Build a Pydantic model from a JSON Schema object (for use as tool args)."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields: dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        json_type = field_schema.get("type", "string")
        py_type = _JSON_TO_PYTHON.get(json_type, Any)
        description = field_schema.get("description", "")

        if field_name in required:
            fields[field_name] = (py_type, ...)
        else:
            default = field_schema.get("default", None)
            fields[field_name] = (Optional[py_type], default)

    return create_model(f"{name}Args", **fields)


# â”€â”€ MCP tool â†’ LangChain BaseTool â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _mcp_tool_to_langchain(mcp_tool: Any, session: Any) -> BaseTool:
    """Convert a single MCP Tool object to a LangChain StructuredTool."""
    tool_name: str = mcp_tool.name
    tool_description: str = mcp_tool.description or ""
    input_schema: dict = mcp_tool.inputSchema or {}

    args_schema = _json_schema_to_pydantic(tool_name, input_schema)

    async def _acall(**kwargs: Any) -> str:
        result = await session.call_tool(tool_name, arguments=kwargs)
        # MCP result has .content list of TextContent / ImageContent
        parts = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else ""

    def _call(**kwargs: Any) -> str:
        return asyncio.get_event_loop().run_until_complete(_acall(**kwargs))

    return StructuredTool(
        name=tool_name,
        description=tool_description,
        args_schema=args_schema,
        func=_call,
        coroutine=_acall,
    )


# â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def load_mcp_tools(session: Any) -> list[BaseTool]:
    """
    Load all tools from an active MCP ClientSession as LangChain tools.

    Args:
        session: An initialized mcp.ClientSession.

    Returns:
        List of LangChain BaseTool instances ready for bind_tools().
    """
    result = await session.list_tools()
    return [_mcp_tool_to_langchain(t, session) for t in result.tools]


async def mcp_tools_from_stdio(
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> list[BaseTool]:
    """
    Connect to an MCP server over stdio, list its tools, and return them as
    LangChain BaseTool instances.

    The connection is opened, tools are listed, and the connection is closed.
    The returned tools hold a reference to the session â€” keep the session alive
    if you plan to call the tools later.  For long-lived use, manage the
    ClientSession yourself with load_mcp_tools().

    Args:
        command: Executable to launch (e.g. "python", "node").
        args:    Arguments for the command.
        env:     Extra environment variables for the subprocess.

    Returns:
        List of LangChain BaseTool instances.
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        raise ImportError("mcp is required: pip install mcp") from exc

    params = StdioServerParameters(command=command, args=args or [], env=env)

    tools: list[BaseTool] = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

    return tools


async def mcp_tools_from_sse(url: str) -> list[BaseTool]:
    """
    Connect to an MCP server over SSE (HTTP), list its tools, and return them
    as LangChain BaseTool instances.

    Args:
        url: SSE endpoint URL (e.g. "http://localhost:8080/sse").

    Returns:
        List of LangChain BaseTool instances.
    """
    try:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
    except ImportError as exc:
        raise ImportError("mcp is required: pip install mcp") from exc

    tools: list[BaseTool] = []
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

    return tools
