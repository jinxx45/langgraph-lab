# Lab 04 — Email Drafter

**Goal:** the model drafts an email, then the graph **pauses and waits for
you**: approve it, ask for changes, or cancel. Nothing is "sent" until a
human says yes. You can even close the program mid-review and come back
tomorrow.

```
START → draft_email → human_review ──approve──→ send_email → END
             ↑             │  ╎
             └──revise─────┘  ╎──cancel──→ END
                              ╎
                    (pauses here, waits for you)
```

## New ideas

**1. `interrupt()` — pause the graph and ask a human.**

```python
decision = interrupt({"subject": ..., "body": ...})
```

The first time this line runs, the graph **stops** and hands the payload to
the caller (`result["__interrupt__"]`). Later, the caller resumes with
`Command(resume=...)`, and *that* value becomes `decision`.

**2. Checkpointers — the graph's save file.** A paused graph has to be
saved somewhere. After every step, a **checkpointer** saves the full state:

```python
graph = builder.compile(checkpointer=SqliteSaver(...))  # on disk, survives restarts
graph = builder.compile(checkpointer=InMemorySaver())   # in RAM, for tests
```

No checkpointer → `interrupt()` can't work, because there's nothing to
resume *from*.

**3. Threads — one saved conversation each.** Every run gets a `thread_id`:

```python
config = {"configurable": {"thread_id": "a1b2c3"}}
graph.invoke({"request": "..."}, config)            # starts thread a1b2c3
graph.invoke(Command(resume=decision), config)      # continues the SAME thread
```

Ten users drafting ten emails = ten threads, each with its own saved state.
`graph.get_state(config)` shows a thread's current values and what it's
waiting for.

**4. `Command(goto=...)` — a node decides where to go next.** Until now,
routing lived in a separate function on a conditional edge. A node can
also return a `Command` that updates the state **and** picks the next node:

```python
return Command(goto="draft_email", update={"feedback": review.feedback})
```

Use it when the routing decision is made *inside* the node's own logic —
like here, after reading the human's answer.

## The gotcha: the paused node runs again

When you resume, LangGraph **re-runs `human_review` from its first line**;
this time `interrupt()` returns your answer instead of pausing. So:

> Never put side effects (sending, saving, charging) **before**
> `interrupt()` in the same node — they'd happen twice.

That's why sending lives in its own node, `send_email`, which only runs
after approval. And it's **idempotent**: the file name is a hash of the
content, so a retry overwrites the same file instead of "sending" twice.

## Production details in this lab

| Problem | What we do |
|---|---|
| Human input is untrusted too | Validated with Pydantic (`Review`); invalid → asked again, not crashed |
| "Revise" with no feedback | Asked again: the model can't improve without knowing what to change |
| Endless revision loop | `MAX_DRAFTS = 4`, then only approve or cancel |
| The process dies mid-review | SQLite checkpointer: resume with `--thread <id>` |
| Side effect runs twice | Sending only after approval, in its own idempotent node |

Our "send" writes a file to `data/outbox/`. Swapping in real email
(SMTP, Gmail API) would change only `send_email` — the review flow stays.

## Run it

```bash
uv run pytest tests/test_lab04.py -v
uv run python -m labs.lab04_email_drafter.graph "Ask my manager for leave this Friday"
```

At the prompt, try `r` to revise, then `q` to quit. Run the command it
prints (`--thread ...`) — the same draft is waiting for you.

## Think about it

- Where else should a human approve before an agent acts? (Payments,
  deleting data, posting publicly...)
- What if two people review the same thread at once?
- Our checkpoints are never deleted. What would you keep, and for how long?
