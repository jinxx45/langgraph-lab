# Lab 03 — Research Helper

**Goal:** answer questions by letting the model **decide** which tools to
call — search Wikipedia, read an article, do maths, check today's date —
and loop until it has enough to answer.

```
START → agent ──(tool calls?)──→ tools
          ↑  │                     │
          │  └──(no: answer)──→ END│
          └────────────────────────┘
```

In Labs 01 and 02, *you* drew every path. Here the model chooses the path at
run time. That's what makes it an **agent**: same graph, but a question
about maths takes a different route than one about history.

## New ideas

**1. Tools are plain functions with a docstring.**

```python
@tool
def calculator(expression: str) -> str:
    """Evaluate arithmetic like '(2+3)*4'. Only numbers, + - * / % ** and parentheses."""
```

`@tool` turns the name, type hints and docstring into a schema the model
sees. Unlike Lab 02's `Field` descriptions, tool docstrings **do** reach the
model — they're sent as tool definitions. The docstring is the tool's
prompt: write it for the model, say *when* to use it.

**2. `bind_tools` — offering tools to the model.** `llm.bind_tools(TOOLS)`
doesn't run anything. It tells the model "these exist". The model replies
with either text (an answer) or `tool_calls` (a request):

```python
AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "2+3"}, "id": "call_1"}])
```

The model never runs code. **Your graph runs the tool** and sends the
result back as a `ToolMessage`.

**3. `MessagesState` and reducers.** Until now, a node's return value
*replaced* a key. Here the state is a conversation, and nodes *append*:

```python
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

`add_messages` is a **reducer**: a function that says how an update merges
into the state. Return `{"messages": [new_msg]}` and it's appended, not
overwritten. Every message — question, tool calls, tool results, answer —
stays in order, so the model sees the whole history on each turn.

**4. `ToolNode` — the prebuilt tool runner.** It reads the last
`AIMessage`, runs each requested tool, and returns one `ToolMessage` per
call. (`tools_condition` is the matching prebuilt router; we write our own
`route_after_agent` so you can see there's no magic.)

## Production details in this lab

| Problem | What we do |
|---|---|
| Model loops forever calling tools | `MAX_TOOL_ROUNDS`: after 5 rounds the agent is called **without** tools, so it has to answer |
| A tool fails (network down, bad maths) | `handle_tool_errors=(...)`: the error goes back to the model as a `ToolMessage` so it can try something else. Only *expected* errors — real bugs still crash loudly |
| `eval()` on model output | Never. `calculator` parses the expression with `ast` and allows only numbers and arithmetic |
| A slow website hangs the agent | Every HTTP call has `timeout=10` |
| Wikipedia's API rules | A descriptive `User-Agent`, as their policy requires |

By default `ToolNode` only catches bad *arguments* and re-raises everything
else — one flaky request would crash the whole run. That's why
`handle_tool_errors` is set explicitly.

## Run it

```bash
uv run pytest tests/test_lab03.py -v
uv run python -m labs.lab03_research_helper.graph "How many years ago was the Taj Mahal completed?"
```

Watch the printed messages: each tool call, each result, then the answer.

## Think about it

- Ask something that needs *two* tools (search, then calculator). Does the
  model chain them on its own?
- Change a docstring to be vague ("Does stuff with numbers"). Does the
  model still pick the right tool?
- The model chose to call a tool — but was the tool's answer actually used
  in its final reply?
