# LangGraph Lab

Learning [LangGraph](https://langchain-ai.github.io/langgraph/) by building
five small graphs, one concept at a time. Everything runs locally and free
on [Ollama](https://ollama.com).

## Labs

| #  | Lab                    | Concept                                    | Status |
|----|------------------------|--------------------------------------------|--------|
| 01 | Joke rater             | State, nodes, edges, loops, context        | ✅     |
| 02 | Support ticket router  | Branching, structured output, retries      | ✅     |
| 03 | Research helper        | Tools, reducers, the agent loop            | ✅     |
| 04 | Email drafter          | Human-in-the-loop, checkpoints, threads    | ✅     |
| 05 | Chat that remembers    | Persistence and long-term memory           | ⏳     |

## How each lab works

1. **Read** the lab's `README.md` — the concept, in plain words.
2. **Build** — `graph.py`: state, nodes, wiring.
3. **Test** — `tests/` uses fake LLMs, so they are fast and deterministic.
   They check your graph's *logic*, not the model.
4. **Eval** — `evals/` runs the real model over a small dataset and
   measures *quality*. Tests tell you it works; evals tell you how well.

## Setup

```bash
winget install astral-sh.uv
winget install Ollama.Ollama
```

Open a new terminal, then:

```bash
ollama pull qwen2.5:7b
uv sync
```

`uv sync` installs Python itself if needed, creates `.venv`, and installs
the dependencies.

## Commands

```bash
uv run pytest                                # all tests
uv run python -m labs.lab01_joke_rater.graph cats   # run a lab
uv run python -m evals.lab01_eval            # run a lab's evals
```
