# qwen_lc

Qwen browser-automation library with LangChain, LangGraph, LangSmith, and MCP integration.

Drives the [chat.qwen.ai](https://chat.qwen.ai) web UI via Playwright — no API key required.
`ChatQwen` is a drop-in replacement for `ChatOpenAI` in LangChain / LangGraph code.

---

## Contents

1. [Installation](#installation)
2. [Configuration (.env)](#configuration-env)
3. [Basic usage](#basic-usage)
4. [LangChain](#langchain)
5. [LangGraph](#langgraph)
6. [LangSmith](#langsmith)
7. [MCP — expose Qwen as a server](#mcp--expose-qwen-as-a-server)
8. [MCP — use MCP tools inside ChatQwen](#mcp--use-mcp-tools-inside-chatqwen)
9. [Swapping from ChatOpenAI](#swapping-from-chatopenai)

---

## Installation

```bash
pip install playwright pydantic
playwright install chromium

# Optional — only needed for the integrations you use
pip install langchain langchain-core langgraph langsmith mcp
```

The library lives at `d:\projs\app\qwen_lc`.  
Add the parent directory to `PYTHONPATH` or install it as an editable package.

---

## Configuration (.env)

Create `.env` in `d:\projs\app\` (same folder that contains `qwen_lc/`):

```env
# Qwen model to use
MODEL=Qwen3.6-Plus

# Thinking mode: Auto | Thinking | Fast
MODEL_MODE=Auto

# Show the browser window (0 = visible, 1 = headless)
HEADLESS=0

# One or more Qwen accounts — each becomes one parallel browser worker
USER1_LOGIN=you@example.com
USER1_PASSWORD=secret

USER2_LOGIN=other@example.com
USER2_PASSWORD=secret2
```

---

## Basic usage

### Synchronous (scripts, notebooks)

```python
from qwen_lc import qwen_start, qwen_send, qwen_stop

qwen_start()                          # opens browser, logs in, selects model
reply = qwen_send("What is 2 + 2?")
print(reply)
qwen_stop()                           # closes all browser windows
```

### With an image

```python
from qwen_lc import qwen_start, qwen_send_with_image, qwen_stop

qwen_start()
reply = qwen_send_with_image("Describe this image", "screenshot.png")
qwen_stop()
```

### Asynchronous (scripts / tests)

```python
import asyncio
from qwen_lc import QwenPool

async def main():
    async with QwenPool() as pool:
        reply = await pool.send("Explain backpropagation.")
        print(reply)

asyncio.run(main())
```

---

## LangChain

### Plain invocation

```python
from qwen_lc import qwen_start, qwen_stop, ChatQwen
from langchain_core.messages import HumanMessage, SystemMessage

qwen_start()

llm = ChatQwen()                      # model_name defaults to Qwen3.6-Plus
result = llm.invoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
])
print(result.content)

qwen_stop()
```

### Tool calling with `bind_tools`

```python
from qwen_lc import qwen_start, qwen_stop, ChatQwen
from langchain_core.messages import HumanMessage, ToolMessage

def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

qwen_start()

llm = ChatQwen().bind_tools([multiply])
msg = llm.invoke([HumanMessage(content="What is 6 times 7?")])

# msg.tool_calls is a list of structured dicts if the model decided to call a tool
if msg.tool_calls:
    tc = msg.tool_calls[0]
    tool_result = multiply(**tc["args"])

    # Feed the result back for the final answer
    final = llm.invoke([
        HumanMessage(content="What is 6 times 7?"),
        msg,
        ToolMessage(content=str(tool_result), tool_call_id=tc["id"]),
    ])
    print(final.content)

qwen_stop()
```

### Structured output with `with_structured_output`

Useful for router / classifier patterns where you need a typed Pydantic object back:

```python
from pydantic import BaseModel
from qwen_lc import qwen_start, qwen_stop, ChatQwen
from langchain_core.messages import HumanMessage

class Sentiment(BaseModel):
    label: str          # "positive", "negative", or "neutral"
    confidence: float   # 0.0 – 1.0

qwen_start()

classifier = ChatQwen().with_structured_output(Sentiment)
result: Sentiment = classifier.invoke([
    HumanMessage(content="I absolutely loved the movie!")
])
print(result.label, result.confidence)   # positive  0.95

qwen_stop()
```

### LCEL chains

`ChatQwen` is a standard LangChain `Runnable`, so it composes with `|`:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qwen_lc import qwen_start, qwen_stop, ChatQwen

qwen_start()

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Translate to {language}."),
        ("human", "{text}"),
    ])
    | ChatQwen()
    | StrOutputParser()
)

print(chain.invoke({"language": "French", "text": "Hello, world!"}))

qwen_stop()
```

---

## LangGraph

### ReAct agent (recommended)

`create_qwen_agent` wraps `langgraph.prebuilt.create_react_agent` and wires up `ChatQwen` automatically.

```python
from qwen_lc import qwen_start, qwen_stop
from qwen_lc.adapters.langgraph_agent import create_qwen_agent

def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

qwen_start()

agent = create_qwen_agent(tools=[multiply, add])

result = agent.invoke({"messages": [("human", "What is (3 * 4) + 10?")]})
print(result["messages"][-1].content)

qwen_stop()
```

### Streaming

```python
for chunk in agent.stream({"messages": [("human", "What is 6 * 7?")]}):
    # Each chunk is a state delta dict, e.g. {"agent": {"messages": [...]}}
    for node, delta in chunk.items():
        for msg in delta.get("messages", []):
            print(f"[{node}]", msg.content or msg.tool_calls)
```

### System prompt

```python
agent = create_qwen_agent(
    tools=[multiply],
    prompt="You are a math assistant. Always show your working.",
)
```

### Checkpointing (multi-turn memory)

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_qwen_agent(
    tools=[multiply],
    checkpointer=MemorySaver(),
)

cfg = {"configurable": {"thread_id": "chat-1"}}

agent.invoke({"messages": [("human", "My name is Alice.")]}, config=cfg)
result = agent.invoke({"messages": [("human", "What is my name?")]}, config=cfg)
print(result["messages"][-1].content)   # Alice
```

### Human-in-the-loop

```python
agent = create_qwen_agent(
    tools=[multiply],
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],   # pause before every tool call
)

cfg = {"configurable": {"thread_id": "t1"}}
state = agent.invoke({"messages": [("human", "What is 6 * 7?")]}, config=cfg)

# Agent stopped before running the tool — inspect the pending call
pending = state["messages"][-1].tool_calls
print("About to call:", pending)

# Resume (pass None to continue without changes)
result = agent.invoke(None, config=cfg)
print(result["messages"][-1].content)
```

### Custom StateGraph

Use `create_qwen_graph` when you need to add your own nodes or edges:

```python
from qwen_lc.adapters.langgraph_agent import create_qwen_graph
from langgraph.graph.message import MessagesState

graph = create_qwen_graph(tools=[multiply])

# graph is a CompiledStateGraph — extend it before compiling if needed
result = graph.invoke({"messages": [("human", "What is 9 * 9?")]})
print(result["messages"][-1].content)
```

### Multi-agent / supervisor graphs

```python
agent = create_qwen_agent(
    tools=[multiply],
    name="math_agent",            # used by supervisor graphs to route messages
)
```

---

## LangSmith

LangSmith traces every LangChain call automatically once the environment variables are set.

### Quick setup

```python
from qwen_lc.adapters.langsmith import configure_langsmith

# Call this BEFORE qwen_start()
configure_langsmith(
    api_key="ls-...",
    project="my-qwen-project",
)

from qwen_lc import qwen_start, qwen_stop, ChatQwen
from langchain_core.messages import HumanMessage

qwen_start()
llm = ChatQwen()
llm.invoke([HumanMessage(content="Hello!")])   # this call appears in LangSmith
qwen_stop()
```

Alternatively, set environment variables directly:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls-...
export LANGCHAIN_PROJECT=my-project
```

### Named trace blocks

Wrap a section of code in a named run visible as a parent span in LangSmith:

```python
from qwen_lc.adapters.langsmith import langsmith_trace

with langsmith_trace("rag-pipeline", tags=["v2", "prod"]) as run:
    docs = retriever.invoke("query")
    answer = llm.invoke([HumanMessage(content="...")])
```

### Toggle tracing

```python
from qwen_lc.adapters.langsmith import disable_langsmith, is_tracing_enabled

disable_langsmith()                     # turn off for the current process
print(is_tracing_enabled())             # False
```

### Get the current run ID

```python
from qwen_lc.adapters.langsmith import get_current_run_id

with langsmith_trace("my-run"):
    run_id = get_current_run_id()
    print(f"https://smith.langchain.com/runs/{run_id}")
```

---

## MCP — expose Qwen as a server

Any MCP-compatible client (Claude Desktop, Cursor, Continue, Zed …) can connect to
this server and call Qwen as if it were a local function.

### Run from the command line

```bash
# stdio transport (default — used by Claude Desktop, Cursor, etc.)
python -m qwen_lc.adapters.mcp.server

# SSE / HTTP transport (accessible over the network)
python -m qwen_lc.adapters.mcp.server --transport sse --port 8080
```

### Embed in your own process

```python
from qwen_lc.adapters.mcp.server import start_server

start_server()          # calls qwen_start() internally, then blocks
```

### Claude Desktop configuration

Add to `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "qwen": {
      "command": "python",
      "args": ["-m", "qwen_lc.adapters.mcp.server"],
      "env": {
        "PYTHONPATH": "D:\\projs\\app"
      }
    }
  }
}
```

The server exposes two tools:

| Tool | Parameters | Description |
|------|-----------|-------------|
| `chat` | `prompt: str` | Send a text prompt, return the response |
| `chat_with_image` | `prompt: str`, `image_path: str` | Send prompt + local image |

---

## MCP — use MCP tools inside ChatQwen

Load tools from any MCP server and pass them to `ChatQwen` via `bind_tools`:

### stdio server (subprocess)

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from qwen_lc.adapters.mcp.tools import load_mcp_tools
from qwen_lc import qwen_start, qwen_stop, ChatQwen
from langchain_core.messages import HumanMessage

async def main():
    params = StdioServerParameters(
        command="python",
        args=["-m", "my_mcp_server"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

    qwen_start()
    llm = ChatQwen().bind_tools(tools)
    result = llm.invoke([HumanMessage(content="Use my tools to help me.")])
    print(result.content)
    qwen_stop()

asyncio.run(main())
```

### Convenience one-liner (stdio)

```python
import asyncio
from qwen_lc.adapters.mcp.tools import mcp_tools_from_stdio
from qwen_lc import qwen_start, qwen_stop, ChatQwen
from langchain_core.messages import HumanMessage

tools = asyncio.run(
    mcp_tools_from_stdio("python", ["-m", "my_mcp_server"])
)

qwen_start()
llm = ChatQwen().bind_tools(tools)
result = llm.invoke([HumanMessage(content="What can you do?")])
qwen_stop()
```

### SSE / HTTP server

```python
import asyncio
from qwen_lc.adapters.mcp.tools import mcp_tools_from_sse

tools = asyncio.run(mcp_tools_from_sse("http://localhost:8080/sse"))
```

### pdf-reader MCP (installed globally)

If you installed `@sylphx/pdf-reader-mcp` (see project root `~/.claude/mcp.json`):

```python
import asyncio
from qwen_lc.adapters.mcp.tools import mcp_tools_from_stdio
from qwen_lc import qwen_start, qwen_stop, ChatQwen
from langchain_core.messages import HumanMessage

tools = asyncio.run(
    mcp_tools_from_stdio("npx", ["@sylphx/pdf-reader-mcp"])
)

qwen_start()
llm = ChatQwen().bind_tools(tools)
result = llm.invoke([HumanMessage(
    content="Read pages 1-3 of C:/Users/me/report.pdf and summarise them."
)])
print(result.content)
qwen_stop()
```

---

## Swapping from ChatOpenAI

`ChatQwen` accepts the same `model=` parameter as `ChatOpenAI` and silently ignores
OpenAI-specific flags (`temperature`, `use_responses_api`, etc.), so most book
exercises and tutorials work with a single-line change:

```python
# Before
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# After — everything else stays the same
from qwen_lc import ChatQwen
llm = ChatQwen(model="Qwen3.6-Plus")
```

`create_qwen_agent` mirrors the full signature of `create_react_agent`:

```python
# Before
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(llm, tools, prompt="You are helpful.")

# After
from qwen_lc.adapters.langgraph_agent import create_qwen_agent
agent = create_qwen_agent(tools, prompt="You are helpful.")
```
