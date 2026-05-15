"""
langchain_browserlm/adapters/mcp/server.py
================================
MCP server that exposes Qwen as a tool provider.

Any MCP-compatible client (Claude Desktop, Cursor, Continue, Zed, etc.) can
connect to this server and call Qwen as if it were a function.

Usage â€” run as a subprocess (stdio transport, default for MCP):
    python -m langchain_browserlm.adapters.mcp.server

Usage â€” run as a standalone SSE server:
    python -m langchain_browserlm.adapters.mcp.server --transport sse --port 8080

Usage â€” embed in your own process:
    from langchain_browserlm.adapters.mcp.server import create_server, start_server

    qwen_start()              # start the browser pool first
    start_server()            # blocks; Ctrl-C to stop

Claude Desktop config (~/.config/claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "qwen": {
          "command": "python",
          "args": ["-m", "langchain_browserlm.adapters.mcp.server"]
        }
      }
    }

Requires:
    pip install mcp
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from langchain_browserlm.adapters.notebook import _pool, qwen_send, qwen_send_with_image, qwen_start


def create_server(name: str = "qwen-lc"):
    """
    Build and return a FastMCP server instance.

    The server exposes two tools:
      - chat(prompt)                  â†’ text response
      - chat_with_image(prompt, path) â†’ text response
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError("mcp is required: pip install mcp") from exc

    mcp = FastMCP(name)

    @mcp.tool()
    def chat(prompt: str) -> str:
        """
        Send a text prompt to Qwen and return the full response.

        Args:
            prompt: The text message to send to Qwen.
        """
        import langchain_browserlm.adapters.notebook as nb
        if nb._pool is None:
            raise RuntimeError(
                "Qwen pool is not running. "
                "Start the server via start_server(env_path=...) or call qwen_start() first."
            )
        return qwen_send(prompt)

    @mcp.tool()
    def chat_with_image(prompt: str, image_path: str) -> str:
        """
        Send a text prompt with a local image file to Qwen and return the response.

        Args:
            prompt:     The text message to send to Qwen.
            image_path: Absolute path to the image file on the local filesystem.
        """
        import langchain_browserlm.adapters.notebook as nb
        if nb._pool is None:
            raise RuntimeError(
                "Qwen pool is not running. "
                "Start the server via start_server(env_path=...) or call qwen_start() first."
            )
        return qwen_send_with_image(prompt, image_path)

    return mcp


def start_server(
    env_path: Optional[Path] = None,
    name: str = "qwen-lc",
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """
    Start the Qwen browser pool and then run the MCP server (blocking).

    Args:
        env_path:  Path to .env file (default: APP_DIR/.env).
        name:      MCP server name.
        transport: "stdio" (for subprocess clients) or "sse" (for HTTP clients).
        host:      Host for SSE transport.
        port:      Port for SSE transport.
    """
    qwen_start(env_path=env_path)
    mcp = create_server(name=name)

    if transport == "sse":
        mcp.run(transport="sse", host=host, port=port)
    else:
        mcp.run(transport="stdio")


# â”€â”€ CLI entry-point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _main() -> None:
    parser = argparse.ArgumentParser(description="Qwen MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="SSE port (default: 8080)")
    parser.add_argument("--env", default=None, help="Path to .env file")
    args = parser.parse_args()

    start_server(
        env_path=Path(args.env) if args.env else None,
        transport=args.transport,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    _main()
