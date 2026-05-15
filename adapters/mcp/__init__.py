"""
langchain_browserlm/adapters/mcp
====================
Model Context Protocol (MCP) integration.

Two directions:

  mcp.server  â€” expose Qwen as an MCP server so any MCP client (Claude Desktop,
                Cursor, etc.) can use it as a tool provider.

  mcp.tools   â€” load tools from an MCP server and expose them as LangChain
                BaseTool instances so ChatQwen can call them via bind_tools().
"""
