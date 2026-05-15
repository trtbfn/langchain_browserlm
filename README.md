# langchain_browserlm

Browser-automation backend for LangChain / LangGraph / LangSmith / MCP.

Drives web-hosted LLMs (Qwen, Kimi, …) via Playwright and exposes them as
standard `ChatOpenAI` objects — no API key required.

---

## Contents

1. [Installation](#installation)
2. [Configuration (.env)](#configuration-env)
3. [Quick start — ChatLLM](#quick-start--chatllm)
4. [LangGraph — supervisor pattern](#langgraph--supervisor-pattern)
5. [LangChain — tools, structured output, chains](#langchain)
6. [LangGraph — ReAct agent](#langgraph--react-agent)
7. [LangSmith tracing](#langsmith-tracing)
8. [MCP — expose as a server](#mcp--expose-as-a-server)
9. [MCP — use MCP tools](#mcp--use-mcp-tools)
10. [Adding a new provider](#adding-a-new-provider)

---

## Installation

```bash
pip install langchain-browserlm
playwright install chromium
```

Or from source:

```bash
pip install git+https://github.com/trtbfn/langchain_browserlm.git
playwright install chromium
```

---

## Configuration (.env)

```env
# One or more accounts — each becomes one parallel browser worker
USER1_LOGIN=you@example.com
USER1_PASSWORD=secret

USER2_LOGIN=other@example.com
USER2_PASSWORD=secret2

# Optional — used by QwenPool / qwen_start() defaults
MODEL=Qwen3.6-Plus
MODEL_MODE=Auto
HEADLESS=0
```

---

## Quick start — ChatLLM

`ChatLLM` is a `ChatOpenAI` subclass that auto-starts a local browser server.
Pass the provider class as the first argument; everything else is standard LangChain.

```python
from langchain_browserlm import ChatLLM
from langchain_browserlm.providers.qwen.provider import QwenProvider

llm = ChatLLM(QwenProvider, model="Qwen3.6-Plus", mode="Fast", env_path=".env")

# Plain ChatOpenAI from here
result = llm.invoke("What is the capital of France?")
print(result.content)
```

---

## LangGraph — supervisor pattern

```python
from langchain_browserlm import ChatLLM
from langchain_browserlm.providers.qwen.provider import QwenProvider
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

worker     = ChatLLM(QwenProvider, model="Qwen3.6-Plus", mode="Fast",     env_path=".env")
supervisor = ChatLLM(QwenProvider, model="QwQ-32B",      mode="Thinking", env_path=".env")

math_agent = create_react_agent(worker, tools=[multiply, add], name="math_agent")
graph = create_supervisor([math_agent], model=supervisor)

result = graph.invoke({"messages": [("human", "What is (3 * 4) + 10?")]})
print(result["messages"][-1].content)
```

---

## LangChain

### Tool calling

```python
from langchain_browserlm import ChatLLM
from langchain_browserlm.providers.qwen.provider import QwenProvider
from langchain_core.messages import HumanMessage, ToolMessage

def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

llm = ChatLLM(QwenProvider, model="Qwen3.6-Plus", mode="Fast", env_path=".env")
llm_with_tools = llm.bind_tools([multiply])

msg = llm_with_tools.invoke([HumanMessage(content="What is 6 times 7?")])
if msg.tool_calls:
    tc = msg.tool_calls[0]
    result = multiply(**tc["args"])
    final = llm_with_tools.invoke([
        HumanMessage(content="What is 6 times 7?"),
        msg,
        ToolMessage(content=str(result), tool_call_id=tc["id"]),
    ])
    print(final.content)
```

### Structured output

```python
from pydantic import BaseModel
from langchain_browserlm import ChatLLM
from langchain_browserlm.providers.qwen.provider import QwenProvider

class Sentiment(BaseModel):
    label: str        # "positive", "negative", or "neutral"
    confidence: float

llm = ChatLLM(QwenProvider, model="Qwen3.6-Plus", env_path=".env")
classifier = llm.with_structured_output(Sentiment)
result: Sentiment = classifier.invoke("I absolutely loved the movie!")
print(result.label, result.confidence)
```

### LCEL chains

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_browserlm import ChatLLM
from langchain_browserlm.providers.qwen.provider import QwenProvider

llm = ChatLLM(QwenProvider, model="Qwen3.6-Plus", env_path=".env")

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Translate to {language}."),
        ("human", "{text}"),
    ])
    | llm
    | StrOutputParser()
)

print(chain.invoke({"language": "French", "text": "Hello, world!"}))
```

---

## LangGraph — ReAct agent

```python
from langchain_browserlm import ChatLLM
from langchain_browserlm.providers.qwen.provider import QwenProvider
from langgraph.prebuilt import create_react_agent

def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

llm = ChatLLM(QwenProvider, model="Qwen3.6-Plus", mode="Fast", env_path=".env")
agent = create_react_agent(llm, tools=[multiply])

result = agent.invoke({"messages": [("human", "What is 6 * 7?")]})
print(result["messages"][-1].content)
```

### Checkpointing (multi-turn memory)

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_react_agent(llm, tools=[multiply], checkpointer=MemorySaver())
cfg = {"configurable": {"thread_id": "chat-1"}}

agent.invoke({"messages": [("human", "My name is Alice.")]}, config=cfg)
result = agent.invoke({"messages": [("human", "What is my name?")]}, config=cfg)
print(result["messages"][-1].content)   # Alice
```

---

## LangSmith tracing

```python
from langchain_browserlm.adapters.langsmith import configure_langsmith

configure_langsmith(api_key="ls-...", project="my-project")
# All subsequent ChatLLM / ChatOpenAI calls are traced automatically
```

Or via environment variables:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls-...
export LANGCHAIN_PROJECT=my-project
```

---

## MCP — expose as a server

```bash
# stdio (Claude Desktop, Cursor, etc.)
python -m langchain_browserlm.adapters.mcp.server

# SSE / HTTP
python -m langchain_browserlm.adapters.mcp.server --transport sse --port 8080
```

Claude Desktop config (`%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "browserlm": {
      "command": "python",
      "args": ["-m", "langchain_browserlm.adapters.mcp.server"]
    }
  }
}
```

---

## MCP — use MCP tools

```python
import asyncio
from langchain_browserlm import ChatLLM
from langchain_browserlm.providers.qwen.provider import QwenProvider
from langchain_browserlm.adapters.mcp.tools import mcp_tools_from_stdio

tools = asyncio.run(mcp_tools_from_stdio("python", ["-m", "my_mcp_server"]))

llm = ChatLLM(QwenProvider, model="Qwen3.6-Plus", env_path=".env")
result = llm.bind_tools(tools).invoke("What can you do?")
print(result.content)
```

---

## Adding a new provider

Subclass `BaseBrowserProvider`, implement `server_command()` plus the three
async UI methods, then pass the class to `ChatLLM`:

```python
import sys
from langchain_browserlm.providers.base import BaseBrowserProvider

class KimiProvider(BaseBrowserProvider):
    @classmethod
    def server_command(cls, port, model, mode, env_path, log_level):
        cmd = [sys.executable, "-m", "kimi_lc.server", "--port", str(port)]
        if env_path:
            cmd += ["--env", env_path]
        return cmd

    async def setup(self, page, cred): ...
    async def send(self, page, request): ...
    async def recover(self, page): ...

    @property
    def label(self): return f"Kimi/{self._model}"

llm = ChatLLM(KimiProvider, model="kimi-k2", env_path=".env")
```
