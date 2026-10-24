# Lab 01 — Joke Rater

**Goal:** write a joke, have a judge score it, and retry until it's good
enough or we run out of attempts.

```
START → write_joke → rate_joke ──(score ≥ 7 or 3 attempts)──→ END
             ↑            │
             └──(retry)───┘
```

## The five ideas

**1. State** — one shared dictionary that flows through the graph. Every
node reads it and returns *only the keys it changed*. LangGraph merges those
updates back in.

```python
class JokeState(TypedDict, total=False):
    topic: str      # input
    joke: str       # written by write_joke
    score: int      # written by rate_joke
    feedback: str   # written by rate_joke
    attempts: int   # written by write_joke
```

**2. Nodes** — plain Python functions: `state in → partial update out`.

```python
def write_joke(state) -> dict:
    ...
    return {"joke": "...", "attempts": 1}   # not the whole state
```

**3. Edges** — fixed arrows: "after `write_joke`, always run `rate_joke`".

**4. Conditional edges** — a function looks at the state and returns the
*name* of the next step. That's how you build a loop:

```python
def should_retry(state) -> str:
    return "done" or "retry"
```

**5. Runtime context** — things every node needs but that aren't *data*:
models, database connections, API clients. They go in a separate
`context`, not in the state.

```python
@dataclass
class Models:
    writer: Runnable
    judge: Runnable

builder = StateGraph(JokeState, context_schema=Models)

def write_joke(state: JokeState, runtime: Runtime[Models]) -> dict:
    reply = runtime.context.writer.invoke(prompt)

graph.invoke({"topic": "cats"}, context=Models(writer=..., judge=...))
```

| State                         | Context                          |
|-------------------------------|----------------------------------|
| Data that changes as it runs  | Fixed for the whole run          |
| Nodes read *and* update it    | Nodes only read it               |
| Saved by checkpoints (Lab 04) | Never saved                      |
| `topic`, `joke`, `score`      | `writer`, `judge`                |

This is dependency injection, LangGraph-style. Tests pass fake models in the
context; the real run passes Ollama. Same graph, no code changes.

## Two design choices worth noticing

- **Nodes are plain top-level functions.** You can test one on its own
  (see the `should_retry` tests) without running the whole graph.
- **The judge returns structured output** (`Rating` with `score` and
  `feedback`), not free text. Never parse "I'd give it a 7/10!" with
  regex — ask the model for a typed object.

## The nodes (in `graph.py`)

1. `write_joke` — prompts with the topic; on retries it adds the previous
   joke and the critic's feedback. Returns the joke and `attempts + 1`.
2. `rate_joke` — sends `judge_prompt(...)` to the judge, copies `score` and
   `feedback` out of the `Rating` into the state.
3. `should_retry` — `"done"` if the score passed or attempts ran out,
   otherwise `"retry"`. Pure logic, so it's tested without any model.

Details worth noticing:
- `state.get("attempts", 0)` and `state.get("feedback")` — those keys don't
  exist on the first pass (`total=False`).
- `judge_prompt` is its own function so the evals rate jokes with exactly
  the same prompt the graph uses.
- The judge prompt says "most jokes deserve 4-6". Remove that line and run
  the evals — watch the pass rate jump. That's prompt calibration.

## Check your work

```bash
uv run pytest tests/test_lab01.py -v      # logic, with fake models
uv run python -m labs.lab01_joke_rater.graph cats   # the real thing
uv run python -m evals.lab01_eval          # quality, over 8 topics
```

## Think about it (after the evals)

- The same model writes *and* judges its own jokes. Is that fair? What would
  a better judge be?
- If the pass rate is 100% on the first attempt, is the judge too easy?
- What stops this graph from looping forever? What happens if you delete it?
