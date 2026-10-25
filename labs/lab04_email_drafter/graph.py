"""Lab 04 — Email Drafter: draft → pause for human review → revise, send or cancel.

Run it:     uv run python -m labs.lab04_email_drafter.graph "Ask my manager for leave on Friday"
Resume it:  uv run python -m labs.lab04_email_drafter.graph --thread <id>
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, RetryPolicy, interrupt
from pydantic import BaseModel, ValidationError

MODEL = "qwen2.5:7b"
MAX_DRAFTS = 4  # the first draft + 3 revisions
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# --- State -----------------------------------------------------------------

class EmailState(TypedDict, total=False):
    request: str        # what the email should do, in the user's words
    subject: str
    body: str
    drafts: int         # how many drafts have been written
    feedback: str       # the reviewer's latest change request
    review_error: str   # why the last review answer was rejected
    status: Literal["sent", "cancelled"]
    sent_to: str        # outbox file, once sent


# --- Structured data: from the model, and from the human --------------------

class Email(BaseModel):
    subject: str
    body: str


class Review(BaseModel):
    # Human input is untrusted too: validate it like model output.
    action: Literal["approve", "revise", "cancel"]
    feedback: str = ""


# --- Context ---------------------------------------------------------------

@dataclass
class Context:
    writer: Runnable  # .invoke(prompt) -> Email
    outbox: Path


# --- Nodes -----------------------------------------------------------------

def draft_prompt(state: EmailState) -> str:
    prompt = (
        "Write a short, polite, professional email.\n"
        "Give a subject line and a body. The body starts with a greeting and ends "
        "with a sign-off. Do not repeat the subject inside the body. "
        "Use [Name] where the sender's name goes.\n\n"
        f"What the email should do:\n<request>\n{state['request']}\n</request>"
    )
    if feedback := state.get("feedback"):
        prompt += (
            f"\n\nYour previous draft:\nSubject: {state['subject']}\n{state['body']}\n\n"
            f"Rewrite it based on this feedback:\n<feedback>\n{feedback}\n</feedback>"
        )
    return prompt


def draft_email(state: EmailState, runtime: Runtime[Context]) -> dict:
    email: Email = runtime.context.writer.invoke(draft_prompt(state))
    return {"subject": email.subject, "body": email.body, "drafts": state.get("drafts", 0) + 1}


def human_review(
    state: EmailState,
) -> Command[Literal["draft_email", "send_email", "human_review", "__end__"]]:
    # On resume this node runs again from the top and interrupt() returns the
    # human's answer. So nothing with side effects may happen before it.
    answer = interrupt(
        {
            "subject": state["subject"],
            "body": state["body"],
            "revisions_left": MAX_DRAFTS - state["drafts"],
            "error": state.get("review_error", ""),
        }
    )

    try:
        review = Review.model_validate(answer)
    except ValidationError:
        return ask_again("Answer with approve, revise or cancel.")

    if review.action == "approve":
        return Command(goto="send_email", update={"review_error": ""})
    if review.action == "cancel":
        return Command(goto=END, update={"status": "cancelled", "review_error": ""})
    if state["drafts"] >= MAX_DRAFTS:
        return ask_again("No revisions left: approve or cancel.")
    if not review.feedback.strip():
        return ask_again("Say what to change, so the next draft can be better.")
    return Command(goto="draft_email", update={"feedback": review.feedback, "review_error": ""})


def ask_again(reason: str) -> Command:
    # Loop back to the same review step; the reason is shown with the draft.
    return Command(goto="human_review", update={"review_error": reason})


def send_email(state: EmailState, runtime: Runtime[Context]) -> dict:
    # Idempotent: the file name comes from the content, so running this twice
    # (a retry, a crash right after writing) overwrites instead of duplicating.
    text = f"Subject: {state['subject']}\n\n{state['body']}\n"
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    runtime.context.outbox.mkdir(parents=True, exist_ok=True)
    path = runtime.context.outbox / f"{digest}.txt"
    path.write_text(text, encoding="utf-8")
    return {"status": "sent", "sent_to": str(path)}


# --- Wiring ----------------------------------------------------------------

builder = StateGraph(EmailState, context_schema=Context)
builder.add_node("draft_email", draft_email, retry_policy=RetryPolicy(max_attempts=3))
builder.add_node("human_review", human_review)
builder.add_node("send_email", send_email)
builder.add_edge(START, "draft_email")
builder.add_edge("draft_email", "human_review")
builder.add_edge("send_email", END)
# No edges out of human_review: it routes itself with Command(goto=...).


def build_graph(checkpointer: BaseCheckpointSaver):
    # interrupt() needs a checkpointer: the paused state has to be saved somewhere.
    return builder.compile(checkpointer=checkpointer)


def default_context() -> Context:
    from langchain_ollama import ChatOllama

    writer = ChatOllama(model=MODEL, temperature=0.4).with_structured_output(Email)
    return Context(writer=writer, outbox=DATA_DIR / "outbox")


# --- Command line ------------------------------------------------------------

def ask_human(payload: dict) -> dict | None:
    """Show the draft, return the review answer, or None to pause."""
    print("\n" + "=" * 60)
    print(f"Subject: {payload['subject']}\n\n{payload['body']}")
    print("=" * 60)
    if payload["error"]:
        print(f"! {payload['error']}")

    choice = input(f"[a]pprove  [r]evise ({payload['revisions_left']} left)  [c]ancel  [q]uit for now > ")
    # PowerShell puts a byte-order mark in front of piped input; ignore it.
    choice = choice.lstrip("﻿").strip().lower()[:1]
    if choice == "q":
        return None
    if choice == "r":
        return {"action": "revise", "feedback": input("What should change? > ")}
    return {"action": {"a": "approve", "c": "cancel"}.get(choice, choice)}


def main() -> None:
    import argparse
    from uuid import uuid4

    from langgraph.checkpoint.sqlite import SqliteSaver

    parser = argparse.ArgumentParser(description="Draft an email with human review.")
    parser.add_argument("request", nargs="*", help="what the email should do")
    parser.add_argument("--thread", help="resume a paused draft by its thread id")
    args = parser.parse_args()
    if not args.request and not args.thread:
        parser.error("give a request, or --thread to resume one")

    DATA_DIR.mkdir(exist_ok=True)
    thread_id = args.thread or uuid4().hex[:8]
    config = {"configurable": {"thread_id": thread_id}}
    context = default_context()

    with SqliteSaver.from_conn_string(str(DATA_DIR / "lab04_checkpoints.db")) as saver:
        graph = build_graph(saver)
        if not args.thread:
            graph.invoke({"request": " ".join(args.request)}, config, context=context)

        while pending := graph.get_state(config).interrupts:
            answer = ask_human(pending[0].value)
            if answer is None:
                print(f"\nPaused. Resume any time with:\n  uv run python -m labs.lab04_email_drafter.graph --thread {thread_id}")
                return
            graph.invoke(Command(resume=answer), config, context=context)

        final = graph.get_state(config).values
        if not final:
            print(f"No draft found for thread {thread_id}.")
        elif final.get("status") == "sent":
            print(f"\nSent → {final['sent_to']}")
        else:
            print(f"\nStatus: {final.get('status', 'unknown')}")


if __name__ == "__main__":
    main()
