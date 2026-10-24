"""Lab 01 — Joke Rater: write → rate → retry until good enough.

Run it:  uv run python -m labs.lab01_joke_rater.graph cats
"""

from dataclasses import dataclass
from typing import TypedDict

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

MODEL = "qwen2.5:7b"
PASS_SCORE = 7
MAX_ATTEMPTS = 3


# --- State: what flows through the graph -----------------------------------

class JokeState(TypedDict, total=False):
    topic: str
    joke: str
    score: int
    feedback: str
    attempts: int


# --- Structured output: what the judge must return -------------------------

class Rating(BaseModel):
    score: int = Field(ge=1, le=10, description="How funny the joke is, 1-10")
    feedback: str = Field(description="One sentence on how to make it funnier")


# --- Context: dependencies handed to every node at run time ----------------
# Not part of the state: it never changes during a run and is never saved.
# Tests pass fake models here; the real run passes Ollama.

@dataclass
class Models:
    writer: Runnable  # chat model: .invoke(prompt) -> message with .content
    judge: Runnable   # structured model: .invoke(prompt) -> Rating


# --- Nodes -----------------------------------------------------------------

def judge_prompt(topic: str, joke: str) -> str:
    # Without the calibration line, small models rate almost everything 8+.
    return (
        f"You are a tough comedy critic. Rate this joke about {topic} from 1 to 10.\n"
        "Most jokes deserve 4-6. Give 8 or more only if it is genuinely surprising.\n\n"
        f"Joke: {joke}"
    )


def write_joke(state: JokeState, runtime: Runtime[Models]) -> dict:
    prompt = f"Write one short, original joke about {state['topic']}. Reply with only the joke."
    # First attempt has no feedback yet; later attempts learn from the critic.
    if feedback := state.get("feedback"):
        prompt += (
            f"\n\nYour previous joke was:\n{state['joke']}\n\n"
            f"A critic said: {feedback}\nWrite a better one."
        )

    reply = runtime.context.writer.invoke(prompt)
    return {"joke": reply.content.strip(), "attempts": state.get("attempts", 0) + 1}


def rate_joke(state: JokeState, runtime: Runtime[Models]) -> dict:
    rating: Rating = runtime.context.judge.invoke(judge_prompt(state["topic"], state["joke"]))
    return {"score": rating.score, "feedback": rating.feedback}


def should_retry(state: JokeState) -> str:
    if state["score"] >= PASS_SCORE or state["attempts"] >= MAX_ATTEMPTS:
        return "done"
    return "retry"


# --- Wiring ----------------------------------------------------------------

builder = StateGraph(JokeState, context_schema=Models)
builder.add_node("write_joke", write_joke)
builder.add_node("rate_joke", rate_joke)
builder.add_edge(START, "write_joke")
builder.add_edge("write_joke", "rate_joke")
builder.add_conditional_edges(
    "rate_joke", should_retry, {"retry": "write_joke", "done": END}
)
graph = builder.compile()


def default_models() -> Models:
    from langchain_ollama import ChatOllama

    return Models(
        writer=ChatOllama(model=MODEL, temperature=0.9),  # creative
        judge=ChatOllama(model=MODEL, temperature=0).with_structured_output(Rating),  # consistent
    )


if __name__ == "__main__":
    import sys

    topic = " ".join(sys.argv[1:]) or "programmers"
    # stream_mode="updates" prints what each node changed, step by step.
    for update in graph.stream(
        {"topic": topic}, context=default_models(), stream_mode="updates"
    ):
        print(update, "\n")
