"""Lab 03 — Research Helper: an agent that decides which tools to call.

Run it:  uv run python -m labs.lab03_research_helper.graph "How old is the Eiffel Tower?"
"""

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy

from labs.lab03_research_helper.tools import EXPECTED_TOOL_ERRORS, TOOLS

MODEL = "qwen2.5:7b"
MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT = (
    "You are a careful research assistant. Look facts up with your tools instead of "
    "guessing, and use the calculator for any arithmetic. When you have enough "
    "information, answer in a few sentences and name the Wikipedia articles you used."
)
OUT_OF_BUDGET = "You have used all your tool calls. Answer now with what you already know."


# --- Context ---------------------------------------------------------------

@dataclass
class Models:
    llm: BaseChatModel  # must support .bind_tools()


# --- Nodes -----------------------------------------------------------------
# State is MessagesState: {"messages": [...]}, and returning {"messages": [msg]}
# APPENDS msg (the add_messages reducer) instead of replacing the list.

def tool_rounds(messages: list[AnyMessage]) -> int:
    return sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)


def agent(state: MessagesState, runtime: Runtime[Models]) -> dict:
    messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
    if tool_rounds(state["messages"]) >= MAX_TOOL_ROUNDS:
        # No tools offered, so the model can't ask for another call: it must answer.
        llm = runtime.context.llm
        messages.append(SystemMessage(OUT_OF_BUDGET))
    else:
        llm = runtime.context.llm.bind_tools(TOOLS)
    return {"messages": [llm.invoke(messages)]}


def route_after_agent(state: MessagesState) -> str:
    # Same job as langgraph.prebuilt.tools_condition, written out.
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "done"


# --- Wiring ----------------------------------------------------------------

builder = StateGraph(MessagesState, context_schema=Models)
builder.add_node("agent", agent, retry_policy=RetryPolicy(max_attempts=3))
builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=EXPECTED_TOOL_ERRORS))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "done": END})
builder.add_edge("tools", "agent")  # the loop: results go back to the model
graph = builder.compile()


def default_models() -> Models:
    from langchain_ollama import ChatOllama

    return Models(llm=ChatOllama(model=MODEL, temperature=0))


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "How many years ago was the Taj Mahal completed?"
    for update in graph.stream(
        {"messages": [("human", question)]}, context=default_models(), stream_mode="updates"
    ):
        for node_output in update.values():
            for message in node_output["messages"]:
                message.pretty_print()
