"""Lab 03 tests: the agent loop with a scripted model, and each tool on its own.
No network: Wikipedia calls are replaced with monkeypatch."""

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from labs.lab03_research_helper import tools
from labs.lab03_research_helper.graph import MAX_TOOL_ROUNDS, OUT_OF_BUDGET, Models, graph
from labs.lab03_research_helper.tools import calculator, read_wikipedia, search_wikipedia


class ScriptedLLM:
    """Plays back AIMessages in order and records each call, including
    whether tools were offered (bind_tools) for that call."""

    def __init__(self, *replies: AIMessage):
        self.replies = list(replies)
        self.calls = []  # (with_tools, messages)

    def bind_tools(self, tool_list):
        return _Bound(self)

    def invoke(self, messages, with_tools=False):
        self.calls.append((with_tools, list(messages)))
        return self.replies.pop(0)


class _Bound:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, messages):
        return self.llm.invoke(messages, with_tools=True)


def call(name, call_id="call_1", **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def ask(llm, question="question"):
    result = graph.invoke({"messages": [("human", question)]}, context=Models(llm))
    return result["messages"]


# --- The agent loop --------------------------------------------------------

def test_answers_directly_when_no_tool_is_needed():
    llm = ScriptedLLM(AIMessage("Hello!"))
    messages = ask(llm)
    assert messages[-1].content == "Hello!"
    assert len(llm.calls) == 1
    assert not any(isinstance(m, ToolMessage) for m in messages)


def test_runs_the_tool_and_sends_the_result_back():
    llm = ScriptedLLM(call("calculator", expression="2 + 3"), AIMessage("It's 5."))
    messages = ask(llm)

    tool_msg = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_msg.content == "5"
    assert tool_msg.tool_call_id == "call_1"
    # The second model call sees the tool result in its history.
    assert tool_msg in llm.calls[1][1]
    assert messages[-1].content == "It's 5."


def test_history_is_appended_not_replaced():
    llm = ScriptedLLM(call("calculator", expression="1 + 1"), AIMessage("2"))
    messages = ask(llm, question="what is 1+1")
    kinds = [type(m).__name__ for m in messages]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]


def test_system_prompt_is_sent_but_not_stored():
    llm = ScriptedLLM(AIMessage("ok"))
    messages = ask(llm)
    assert isinstance(llm.calls[0][1][0], SystemMessage)
    assert not any(isinstance(m, SystemMessage) for m in messages)


def test_tool_error_goes_back_to_the_model_instead_of_crashing():
    llm = ScriptedLLM(call("calculator", expression="__import__('os')"), AIMessage("Sorry."))
    messages = ask(llm)
    tool_msg = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_msg.status == "error"
    assert "only numbers" in tool_msg.content
    assert messages[-1].content == "Sorry."


def test_stops_offering_tools_after_the_budget():
    rounds = [call("today", call_id=f"call_{i}") for i in range(MAX_TOOL_ROUNDS)]
    llm = ScriptedLLM(*rounds, AIMessage("Final answer."))
    messages = ask(llm)

    assert len(llm.calls) == MAX_TOOL_ROUNDS + 1
    assert all(with_tools for with_tools, _ in llm.calls[:-1])
    last_with_tools, last_messages = llm.calls[-1]
    assert not last_with_tools
    assert last_messages[-1].content == OUT_OF_BUDGET
    assert messages[-1].content == "Final answer."


# --- calculator ------------------------------------------------------------

@pytest.mark.parametrize(
    ("expression", "expected"),
    [("2 + 3", "5"), ("(2 + 3) * 4", "20"), ("2026 - 1653", "373"), ("-4 + 10 / 4", "-1.5"), ("2 ** 10", "1024")],
)
def test_calculator_does_arithmetic(expression, expected):
    assert calculator.invoke({"expression": expression}) == expected


@pytest.mark.parametrize("expression", ["__import__('os')", "open('x')", "x + 1", "2 ** 1000", "2 +"])
def test_calculator_rejects_anything_else(expression):
    with pytest.raises(ValueError):
        calculator.invoke({"expression": expression})


# --- Wikipedia tools (network replaced) -------------------------------------

def test_search_formats_hits_and_strips_html(monkeypatch):
    fake = {"query": {"search": [{"title": "Taj Mahal", "snippet": 'the <span class="match">Taj</span> Mahal &amp; gardens'}]}}
    monkeypatch.setattr(tools, "_http_get", lambda url, params=None: fake)
    assert search_wikipedia.invoke({"query": "taj"}) == "- Taj Mahal: the Taj Mahal & gardens"


def test_search_with_no_results(monkeypatch):
    monkeypatch.setattr(tools, "_http_get", lambda url, params=None: {"query": {"search": []}})
    assert search_wikipedia.invoke({"query": "zzzz"}) == "No results."


def test_read_returns_the_summary_and_encodes_the_title(monkeypatch):
    seen = {}

    def fake_get(url, params=None):
        seen["url"] = url
        return {"extract": "A mausoleum in Agra."}

    monkeypatch.setattr(tools, "_http_get", fake_get)
    assert read_wikipedia.invoke({"title": "Taj Mahal"}) == "A mausoleum in Agra."
    assert seen["url"].endswith("/Taj_Mahal")
