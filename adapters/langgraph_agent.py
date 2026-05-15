"""
langchain_browserlm/adapters/langgraph_agent.py
=====================================
LangGraph integration helpers for building agents with Qwen.

Usage:
    from langchain_browserlm import qwen_start, qwen_stop
    from langchain_browserlm.adapters.langgraph_agent import create_qwen_agent

    qwen_start()

    def multiply(a: int, b: int) -> int:
        '''Multiply two integers.'''
        return a * b

    agent = create_qwen_agent(tools=[multiply])

    # Invoke synchronously
    result = agent.invoke({"messages": [("human", "What is 6 * 7?")]})
    print(result["messages"][-1].content)

    # Stream (yields state deltas)
    for chunk in agent.stream({"messages": [("human", "What is 6 * 7?")]}):
        print(chunk)

    qwen_stop()

Custom state / checkpointing:
    from langgraph.checkpoint.memory import MemorySaver

    agent = create_qwen_agent(
        tools=[multiply],
        checkpointer=MemorySaver(),
        state_modifier="You are a helpful math assistant.",
    )
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.tools import BaseTool

from langchain_browserlm.adapters.langchain_chat import ChatQwen


def create_qwen_agent(
    tools: Sequence[Any],
    *,
    model_name: str = "Qwen3.6-Plus",
    checkpointer: Any = None,
    prompt: str | Any | None = None,
    state_modifier: str | Any | None = None,
    state_schema: Any = None,
    name: str | None = None,
    pre_model_hook: Any | None = None,
    post_model_hook: Any | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    **kwargs: Any,
):
    """
    Build a LangGraph ReAct agent backed by ChatQwen.

    Mirrors the signature of langgraph.prebuilt.create_react_agent so that
    book exercises using ChatOpenAI can be swapped to ChatQwen with minimal
    changes.

    Args:
        tools:            List of callable tools or LangChain BaseTool instances.
        model_name:       Qwen model name for LangSmith traces.
        checkpointer:     Optional LangGraph checkpointer (e.g. InMemorySaver()).
        prompt:           System prompt string, SystemMessage, or callable.
                          (Preferred â€” newer LangGraph API.)
        state_modifier:   Alias for `prompt` kept for backward compatibility.
        state_schema:     Custom TypedDict state schema (e.g. AgentState).
        name:             Agent name used in multi-agent / supervisor graphs.
        pre_model_hook:   Callable run before the LLM â€” use for guardrails.
        post_model_hook:  Callable run after the LLM.
        interrupt_before: Node names to pause before (for human-in-the-loop).
        interrupt_after:  Node names to pause after.
        **kwargs:         Forwarded to create_react_agent.

    Returns:
        CompiledGraph â€” call .invoke(), .stream(), or .astream() on it.

    Requires:
        pip install langgraph
    """
    try:
        from langgraph.prebuilt import create_react_agent
    except ImportError as exc:
        raise ImportError(
            "langgraph is required: pip install langgraph"
        ) from exc

    llm = ChatQwen(model_name=model_name).bind_tools(tools)

    create_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        create_kwargs["checkpointer"] = checkpointer

    # `prompt` takes precedence over the older `state_modifier` alias.
    effective_prompt = prompt if prompt is not None else state_modifier
    if effective_prompt is not None:
        import inspect
        sig = inspect.signature(create_react_agent)
        if "prompt" in sig.parameters:
            create_kwargs["prompt"] = effective_prompt
        else:
            create_kwargs["state_modifier"] = effective_prompt

    if state_schema is not None:
        create_kwargs["state_schema"] = state_schema
    if name is not None:
        create_kwargs["name"] = name
    if pre_model_hook is not None:
        create_kwargs["pre_model_hook"] = pre_model_hook
    if post_model_hook is not None:
        create_kwargs["post_model_hook"] = post_model_hook
    if interrupt_before is not None:
        create_kwargs["interrupt_before"] = interrupt_before
    if interrupt_after is not None:
        create_kwargs["interrupt_after"] = interrupt_after
    create_kwargs.update(kwargs)

    return create_react_agent(llm, list(tools), **create_kwargs)


def create_qwen_graph(
    tools: Sequence[Any],
    *,
    model_name: str = "Qwen3.6-Plus",
    checkpointer: Any = None,
    state_schema: Any = None,
    **kwargs: Any,
):
    """
    Build a LangGraph StateGraph with a Qwen model node and a ToolNode.

    Prefer create_qwen_agent for most use cases.  Use this when you need
    a custom StateGraph you can extend with additional nodes.

    Args:
        tools:        List of callable tools or LangChain BaseTool instances.
        model_name:   Qwen model name for LangSmith traces.
        checkpointer: Optional LangGraph checkpointer.
        state_schema: Custom TypedDict state schema (default: MessagesState).
        **kwargs:     Forwarded to StateGraph.

    Returns:
        CompiledStateGraph
    """
    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.graph.message import MessagesState
        from langgraph.prebuilt import ToolNode, tools_condition
    except ImportError as exc:
        raise ImportError(
            "langgraph is required: pip install langgraph"
        ) from exc

    schema = state_schema or MessagesState
    llm = ChatQwen(model_name=model_name).bind_tools(tools)
    tool_node = ToolNode(list(tools))

    def call_model(state: Any) -> dict:
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(schema, **kwargs)
    builder.add_node("agent", call_model)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    builder.add_edge("agent", END)

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return builder.compile(**compile_kwargs)
