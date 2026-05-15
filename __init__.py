"""
langchain_browserlm
====================
Browser-automation backend for LangChain / LangGraph / LangSmith / MCP.

Drives real browser sessions (Playwright) to talk to web-hosted LLMs — Qwen,
Kimi, and any future provider — and exposes them as standard ChatOpenAI
objects.  No API keys required; the library logs into the provider's web UI
and scrapes the response.

Layer overview
--------------
  langchain_browserlm                             <- this file, public API
  langchain_browserlm.adapters.langchain_server  <- ChatLLM  (recommended entry point)
  langchain_browserlm.providers.base             <- BaseBrowserProvider ABC
  langchain_browserlm.providers.qwen.provider    <- QwenProvider
  langchain_browserlm.runtime.pool               <- BrowserPool / QwenPool
  langchain_browserlm.server                     <- OpenAI-compatible FastAPI server
  langchain_browserlm.adapters.notebook          <- qwen_start / qwen_send / qwen_stop (sync)
  langchain_browserlm.adapters.langgraph_agent   <- create_qwen_agent / create_qwen_graph
  langchain_browserlm.adapters.langsmith         <- configure_langsmith / langsmith_trace
  langchain_browserlm.adapters.mcp.server        <- MCP server exposing the LLM as a tool
  langchain_browserlm.adapters.mcp.tools         <- load MCP tools for use with ChatLLM

Quick start -- ChatLLM (recommended):
    from langchain_browserlm import ChatLLM
    from langchain_browserlm.providers.qwen.provider import QwenProvider

    llm_worker     = ChatLLM(QwenProvider, model="Qwen3.6-Plus", mode="Fast",     env_path=".env")
    llm_supervisor = ChatLLM(QwenProvider, model="QwQ-32B",      mode="Thinking", env_path=".env")

    # Plain ChatOpenAI from here -- bind_tools(), with_structured_output(), streaming...
    agent = create_react_agent(llm_worker, tools=[search, weather])

Quick start -- LangGraph supervisor pattern:
    from langchain_browserlm import ChatLLM
    from langchain_browserlm.providers.qwen.provider import QwenProvider
    from langgraph_supervisor import create_supervisor

    worker     = ChatLLM(QwenProvider, model="Qwen3.6-Plus", mode="Fast",     env_path=".env")
    supervisor = ChatLLM(QwenProvider, model="QwQ-32B",      mode="Thinking", env_path=".env")

    graph = create_supervisor([worker_agent, ...], model=supervisor)

Quick start -- notebook / script (sync):
    from langchain_browserlm import qwen_start, qwen_send, qwen_stop

    qwen_start()
    reply = qwen_send("Explain backpropagation in simple terms.")
    qwen_stop()

Quick start -- LangSmith tracing:
    from langchain_browserlm.adapters.langsmith import configure_langsmith

    configure_langsmith(api_key="ls-...", project="my-project")
    # all subsequent ChatLLM / ChatOpenAI calls are traced automatically

Quick start -- MCP server:
    python -m langchain_browserlm.adapters.mcp.server          # stdio (default)
    python -m langchain_browserlm.adapters.mcp.server --transport sse --port 8080

Quick start -- async pool:
    from langchain_browserlm import QwenPool

    async with QwenPool() as pool:
        reply = await pool.send("Your prompt here")

Adding a new provider:
    from langchain_browserlm.providers.base import BaseBrowserProvider

    class KimiProvider(BaseBrowserProvider):
        @classmethod
        def server_command(cls, port, model, mode, env_path, log_level):
            return [sys.executable, "-m", "kimi_lc.server", "--port", str(port), ...]
        # implement setup(), send(), recover(), label ...

    llm = ChatLLM(KimiProvider, model="kimi-k2", env_path=".env")
"""

# -- Sync notebook API --------------------------------------------------------
from langchain_browserlm.adapters.notebook import qwen_send, qwen_send_with_image, qwen_start, qwen_stop

# -- Async pool ---------------------------------------------------------------
from langchain_browserlm.runtime.pool import QwenPool

# -- LangChain entry point ----------------------------------------------------
from langchain_browserlm.adapters.langchain_server import ChatLLM

__all__ = [
    # Recommended LangChain entry point
    "ChatLLM",
    # Sync notebook API
    "qwen_start",
    "qwen_send",
    "qwen_send_with_image",
    "qwen_stop",
    # Async pool
    "QwenPool",
]
