# Lab 02 — Support Ticket Router

**Goal:** read a customer support ticket, decide what kind it is, and send it
to the right specialist — or to a human when it's urgent or unclear.

```
                        ┌──→ billing_agent ───┐
                        ├──→ technical_agent ─┤
START → classify ──route┼──→ account_agent ───┼──→ END
                        ├──→ general_agent ───┤
                        └──→ human_handoff ───┘
```

Lab 01 used a conditional edge to **loop**. Here it **branches**: one
decision, five possible paths, each ending the run.

## New ideas

**1. Routing on structured output.** The classifier returns a typed
`Triage` object. `Literal[...]` restricts a field to a fixed set of values —
the model *cannot* invent a category called "payments-ish".

```python
class Triage(BaseModel):
    reasoning: str
    category: Literal["billing", "technical", "account", "general"]
    urgency: Literal["low", "high"]
    confidence: int = Field(ge=0, le=100)
```

Why 0-100 and not 0-1? The first version used `0-1`, and qwen2.5 answered
`"confidence": 100` — Pydantic rejected it and the run crashed. Two lessons:
validation caught bad data *before* it reached the router, and when a model
keeps getting a format wrong, change the format to match how the model
thinks instead of fighting it in the prompt.

`reasoning` comes **first** on purpose: the model writes its explanation
before it commits to a category. Fields are generated in order, so this is a
cheap way to make it "think before answering".

**Getting the format right, the production way.** `with_structured_output`
sends the schema to Ollama, which *forces* the shape: keys, types and
`Literal` choices. It does **not** show the model your `Field`
descriptions or limits like `le=100` — so the full recipe is:

1. Pydantic model = the single source of truth for the format
2. Constrained output enforces the shape (`with_structured_output`)
3. The **prompt** states the meaning: what "high urgency" is, the 0-100 range
4. Pydantic validates every answer
5. Invalid? Retry with the error message, so the model can fix it
   (`MAX_FORMAT_RETRIES`). Still invalid? Hand off to a human.

Two kinds of failure, two tools:

| Failure | Example | Handled by |
|---|---|---|
| Infrastructure | Ollama down, timeout, 5xx | `RetryPolicy` on the node: same call again, with backoff |
| Bad answer | `confidence: 500` | `classify()` itself: retry *with feedback*, then fall back |

`RetryPolicy` skips `ValueError` by default, and both `ValidationError` and
`OutputParserException` are `ValueError`s — so the two never overlap.

**2. The router is plain Python, not an LLM.** `route_ticket` reads the
state and applies business rules — "urgent or unsure goes to a human". The
LLM *classifies*; your code *decides*. Rules you can read, test and change
without touching a prompt.

**3. Specialist nodes with system prompts.** Each agent gets its own
instructions as a **system message** (who you are, what you must never do)
and the ticket as a **human message**:

```python
[("system", "You are a billing specialist. Never promise..."),
 ("human", ticket)]
```

**4. Not every node needs an LLM.** `human_handoff` just returns a fixed
acknowledgement. Cheaper, instant, and impossible to get wrong.

## Evals: now we have ground truth

Lab 01's eval asked a model "is this funny?" — no right answer. A classifier
is different: every ticket in `evals/data/lab02_tickets.jsonl` has a **human
label**, so we can measure exactly:

| Metric | Question it answers |
|---|---|
| Accuracy | How often is the category right? |
| Precision / recall per category | Which categories get confused? |
| Confusion matrix | Confused with *what*? |
| Missed urgent | Urgent tickets that did *not* go to a human — the dangerous error |
| Confidence calibration | When the model says it's unsure, is it actually wrong more often? |

We eval the **classifier on its own**, not the whole graph. The specialists'
replies don't affect routing accuracy, and skipping them makes the eval ~5x
faster. Eval the component whose quality you're measuring.

## Run it

```bash
uv run pytest tests/test_lab02.py -v
uv run python -m labs.lab02_ticket_router.graph "I was charged twice for my plan"
uv run python -m evals.lab02_eval
```

## Think about it

- LLMs are bad at reporting their own confidence. Does the calibration
  number agree?
- Is a missed urgent ticket worse than an unnecessary escalation? How would
  you change `route_ticket` if your support team is small?
- Two tickets in the dataset are genuinely ambiguous. What should the
  "right" answer even be?
