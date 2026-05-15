"""
langchain_browserlm/adapters/langchain_chat.py
===================================
LangChain BaseChatModel backed by the langchain_browserlm browser-automation library.

Supports:
  - LangChain tool calling via bind_tools()
  - LangGraph ReAct agents (tool_calls on AIMessage)
  - LangSmith tracing (automatic via LANGCHAIN_TRACING_V2 env var)
  - Async generation (_agenerate)
  - Streaming interface (_stream / _astream) â€” single-chunk since the browser
    does not expose a streaming API, but compatible with streaming consumers

Usage:
    from langchain_browserlm import qwen_start, qwen_stop
    from langchain_browserlm.adapters.langchain_chat import ChatQwen

    qwen_start()
    llm = ChatQwen().bind_tools([my_tool])
    result = llm.invoke([HumanMessage(content="What is 3 * 7?")])
    qwen_stop()
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict, Field, model_validator

from langchain_browserlm.adapters import notebook as _notebook

# â”€â”€ Tool schema serialisation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


def _tool_schema_line(tool: Any) -> str:
    name = tool.name
    desc = (tool.description or "").strip()
    args = getattr(tool, "args", {})
    args_str = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in args.items())
    return f"- {name}({args_str}): {desc}"


# â”€â”€ Message â†’ flat text â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _messages_to_prompt(messages: List[BaseMessage], tools: list) -> str:
    tool_block = (
        _TOOL_SYSTEM_BLOCK.format(
            schemas="\n".join(_tool_schema_line(t) for t in tools)
        )
        if tools
        else ""
    )

    parts: list[str] = []
    has_system = any(isinstance(m, SystemMessage) for m in messages)

    if tools and not has_system:
        parts.append(tool_block.strip())

    for msg in messages:
        if isinstance(msg, SystemMessage):
            parts.append(f"{msg.content}{tool_block}")

        elif isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            parts.append(f"USER: {content}")

        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                obj = {"name": tc["name"], "args": tc["args"]}
                parts.append(f"ASSISTANT:\n```fn\n{json.dumps(obj)}\n```")
            else:
                parts.append(f"ASSISTANT:\n```ans\n{msg.content}\n```")

        elif isinstance(msg, ToolMessage):
            parts.append(f"TOOL_RESULT[{msg.tool_call_id}]: {msg.content}")

    return "\n\n".join(parts)


# â”€â”€ Response â†’ AIMessage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _parse_response(text: str) -> AIMessage:
    """
    Parse Qwen's raw .phase-answer innerText into an AIMessage.

    The browser renders ```fn\\n{...}\\n``` as:
        fn\\n<line_numbers>\\n<content>
    We strip the tag+numbers prefix then parse the JSON.
    """
    clean = text.replace('\xa0', ' ')
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()

    # fn block (function call)
    m_fn = re.match(r"fn\s+(?:\d+\s+)*(.*)", clean, re.DOTALL)
    if m_fn:
        body = m_fn.group(1).strip()
        try:
            start = body.index("{")
            obj, _ = json.JSONDecoder().raw_decode(body[start:])
            tool_call = {
                "name": obj["name"],
                "args": obj.get("args", obj.get("arguments", {})),
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }
            return AIMessage(content="", tool_calls=[tool_call])
        except (ValueError, KeyError):
            pass

    # ans block (final answer)
    m_ans = re.match(r"ans\s+(?:\d+\s+)*(.*)", clean, re.DOTALL)
    if m_ans:
        return AIMessage(content=m_ans.group(1).strip())

    # Fallback: bare JSON with "type" field (legacy)
    try:
        start = clean.index("{")
        obj, _ = json.JSONDecoder().raw_decode(clean[start:])
        kind = obj.get("type")
        if kind == "tool_call":
            return AIMessage(content="", tool_calls=[{
                "name": obj["name"],
                "args": obj.get("arguments", obj.get("args", {})),
                "id": obj.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "type": "tool_call",
            }])
        if kind == "final":
            return AIMessage(content=obj.get("content", obj.get("answer", "")))
    except (ValueError, KeyError):
        pass

    return AIMessage(content=clean or text)


def _to_chunk(message: AIMessage) -> AIMessageChunk:
    """Convert an AIMessage to a single AIMessageChunk for streaming consumers."""
    if message.tool_calls:
        tc = message.tool_calls[0]
        return AIMessageChunk(
            content="",
            tool_call_chunks=[{
                "name": tc["name"],
                "args": json.dumps(tc["args"]),
                "id": tc["id"],
                "index": 0,
                "type": "tool_call_chunk",
            }],
        )
    return AIMessageChunk(content=message.content)


# â”€â”€ ChatQwen â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ChatQwen(BaseChatModel):
    """
    LangChain chat model powered by the langchain_browserlm browser automation library.

    Drop-in for ChatOpenAI where you need browser-based Qwen access:

        llm = ChatQwen(model_name="Qwen3.6-Plus").bind_tools([multiply])
        result = llm.invoke([HumanMessage(content="What is 3 * 7?")])

    Also accepts ChatOpenAI-style `model=` parameter and silently ignores
    OpenAI-specific flags (use_responses_api, use_previous_response_id, etc.)
    so book exercises that use ChatOpenAI can be switched to ChatQwen with
    minimal changes.

    LangSmith tracing is automatic when LANGCHAIN_TRACING_V2=true and
    LANGCHAIN_API_KEY are set in the environment.

    LangGraph compatibility: bind_tools() + structured tool_calls on AIMessage
    are fully supported â€” use with langgraph.prebuilt.create_react_agent.
    with_structured_output() is supported for router / guardrail patterns.
    """

    # extra='ignore' silently discards OpenAI-specific params like
    # use_responses_api, use_previous_response_id, temperature, etc.
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore", populate_by_name=True)

    model_name: str = Field(default="Qwen3.6-Plus", description="Qwen model name for LangSmith traces.")
    bound_tools: List[Any] = Field(default_factory=list)
    env_path: Optional[str] = Field(default=None, description="Path to .env file. When set, the pool is started automatically on first use.")

    @model_validator(mode="before")
    @classmethod
    def _normalize_params(cls, data: Any) -> Any:
        """Accept `model=` as an alias for `model_name=`."""
        if isinstance(data, dict) and "model" in data and "model_name" not in data:
            data = dict(data)
            data["model_name"] = data.pop("model")
        return data

    def _ensure_started(self) -> None:
        """Start the pool lazily if env_path is set and the pool is not yet running."""
        if _notebook._pool is not None:
            return
        if self.env_path is not None:
            from pathlib import Path
            _notebook.qwen_start(env_path=Path(self.env_path))
        else:
            raise RuntimeError(
                "Qwen pool is not running. Either call qwen_start() first, "
                "or pass env_path= to ChatQwen() for automatic startup."
            )

    @property
    def _llm_type(self) -> str:
        return "qwen-browser"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Metadata surfaced in LangSmith traces."""
        return {
            "model_name": self.model_name,
            "ls_provider": "qwen",
            "ls_model_name": self.model_name,
            "ls_model_type": "chat",
        }

    # â”€â”€ Tool binding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def bind_tools(self, tools: list, **kwargs: Any) -> "ChatQwen":
        """Return a new ChatQwen instance with the tools registered."""
        return ChatQwen(model_name=self.model_name, bound_tools=list(tools), env_path=self.env_path)

    # â”€â”€ Structured output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def with_structured_output(self, schema: Any, *, include_raw: bool = False, **kwargs: Any) -> Any:
        """
        Return a Runnable that parses responses as the given Pydantic schema.

        Enables the router / guardrail patterns from the book:

            llm_router = ChatQwen().with_structured_output(AgentTypeOutput)
            result = llm_router.invoke(messages)   # â†’ AgentTypeOutput instance

        Args:
            schema:      A Pydantic BaseModel class (v1 or v2).
            include_raw: Ignored (kept for API compatibility with ChatOpenAI).
        """
        import json as _json
        from langchain_core.runnables import RunnableLambda

        # Build a compact JSON schema description to inject into the prompt.
        if hasattr(schema, "model_json_schema"):
            schema_str = _json.dumps(schema.model_json_schema(), indent=2)
        elif hasattr(schema, "schema"):
            schema_str = _json.dumps(schema.schema(), indent=2)
        else:
            schema_str = str(schema)

        json_instruction = (
            "You MUST respond with ONLY a valid JSON object and nothing else "
            "(no markdown fences, no explanation, no extra text).\n"
            f"The JSON must match this schema:\n{schema_str}"
        )

        # Use a plain ChatQwen (no bound tools) so the fn/ans protocol is not
        # injected â€” we just want raw JSON back.
        base_llm = ChatQwen(model_name=self.model_name, env_path=self.env_path)

        def _invoke(messages: Any, **call_kwargs: Any) -> Any:
            msgs = list(messages)
            # Append the JSON instruction to the first SystemMessage, or add one.
            augmented: list = []
            injected = False
            for m in msgs:
                if isinstance(m, SystemMessage) and not injected:
                    augmented.append(SystemMessage(content=m.content + "\n\n" + json_instruction))
                    injected = True
                else:
                    augmented.append(m)
            if not injected:
                augmented = [SystemMessage(content=json_instruction)] + augmented

            result = base_llm._generate(augmented)
            raw = result.generations[0].message.content

            # Extract the first JSON object from the response.
            try:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                parsed = _json.loads(raw[start:end])
            except (ValueError, _json.JSONDecodeError):
                # Try stripping markdown code fences
                m_code = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
                if m_code:
                    parsed = _json.loads(m_code.group(1))
                else:
                    parsed = _json.loads(raw.strip())

            if hasattr(schema, "model_validate"):
                return schema.model_validate(parsed)
            if hasattr(schema, "parse_obj"):
                return schema.parse_obj(parsed)
            return parsed

        return RunnableLambda(_invoke)

    # â”€â”€ Core sync generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._ensure_started()
        prompt = _messages_to_prompt(messages, self.bound_tools)
        raw = _notebook.qwen_send(prompt)
        message = _parse_response(raw)
        if run_manager and message.content:
            run_manager.on_llm_new_token(message.content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # â”€â”€ Sync streaming â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Yield a single chunk (browser does not stream, but interface is compatible)."""
        result = self._generate(messages, stop=stop, run_manager=None, **kwargs)
        msg = result.generations[0].message
        chunk = _to_chunk(msg)
        if run_manager and chunk.content:
            run_manager.on_llm_new_token(chunk.content)
        yield ChatGenerationChunk(message=chunk)

    # â”€â”€ Async generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._ensure_started()
        prompt = _messages_to_prompt(messages, self.bound_tools)
        raw = await _notebook._pool.send(prompt)
        message = _parse_response(raw)
        if run_manager and message.content:
            await run_manager.on_llm_new_token(message.content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # â”€â”€ Async streaming â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Yield a single chunk (browser does not stream, but interface is compatible)."""
        result = await self._agenerate(messages, stop=stop, run_manager=None, **kwargs)
        msg = result.generations[0].message
        chunk = _to_chunk(msg)
        if run_manager and chunk.content:
            await run_manager.on_llm_new_token(chunk.content)
        yield ChatGenerationChunk(message=chunk)
