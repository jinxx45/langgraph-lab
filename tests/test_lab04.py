"""Lab 04 tests: pausing for a human, resuming, and surviving a restart."""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from labs.lab04_email_drafter.graph import MAX_DRAFTS, Context, Email, build_graph


class ScriptedWriter:
    """Returns "Draft 1", "Draft 2", ... and records every prompt."""

    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        n = len(self.prompts)
        return Email(subject=f"Draft {n}", body=f"Body {n}")


@pytest.fixture
def writer():
    return ScriptedWriter()


@pytest.fixture
def context(writer, tmp_path):
    return Context(writer=writer, outbox=tmp_path / "outbox")


@pytest.fixture
def graph():
    return build_graph(InMemorySaver())


def config(thread="t1"):
    return {"configurable": {"thread_id": thread}}


def start(graph, context, thread="t1"):
    return graph.invoke({"request": "Ask for Friday off"}, config(thread), context=context)


def resume(graph, context, answer, thread="t1"):
    return graph.invoke(Command(resume=answer), config(thread), context=context)


def pending(result):
    """The payload shown to the human, or None if the graph finished."""
    interrupts = result.get("__interrupt__")
    return interrupts[0].value if interrupts else None


# --- Pausing ---------------------------------------------------------------

def test_pauses_with_the_draft_before_sending(graph, context):
    result = start(graph, context)
    payload = pending(result)
    assert payload["subject"] == "Draft 1"
    assert payload["body"] == "Body 1"
    assert payload["revisions_left"] == MAX_DRAFTS - 1
    assert not context.outbox.exists()  # nothing sent without approval


def test_paused_thread_reports_what_it_waits_for(graph, context):
    start(graph, context)
    snapshot = graph.get_state(config())
    assert snapshot.next == ("human_review",)
    assert snapshot.values["drafts"] == 1


# --- The three answers -----------------------------------------------------

def test_approve_sends_the_email(graph, context):
    start(graph, context)
    result = resume(graph, context, {"action": "approve"})
    assert pending(result) is None
    assert result["status"] == "sent"
    sent = list(context.outbox.iterdir())
    assert len(sent) == 1
    assert sent[0].read_text(encoding="utf-8") == "Subject: Draft 1\n\nBody 1\n"


def test_revise_redrafts_with_the_feedback(graph, context, writer):
    start(graph, context)
    result = resume(graph, context, {"action": "revise", "feedback": "More formal please"})
    assert pending(result)["subject"] == "Draft 2"
    assert "More formal please" in writer.prompts[1]
    assert "Draft 1" in writer.prompts[1]  # the previous draft is shown to the model
    assert "feedback" not in writer.prompts[0].lower()


def test_cancel_sends_nothing(graph, context):
    start(graph, context)
    result = resume(graph, context, {"action": "cancel"})
    assert result["status"] == "cancelled"
    assert not context.outbox.exists()


# --- Bad answers are asked again, not crashed on ----------------------------

def test_invalid_answer_asks_again(graph, context, writer):
    start(graph, context)
    payload = pending(resume(graph, context, {"action": "maybe"}))
    assert payload["subject"] == "Draft 1"
    assert "approve, revise or cancel" in payload["error"]
    assert len(writer.prompts) == 1  # no new draft


def test_revise_without_feedback_asks_again(graph, context, writer):
    start(graph, context)
    payload = pending(resume(graph, context, {"action": "revise", "feedback": "  "}))
    assert "what to change" in payload["error"]
    assert len(writer.prompts) == 1


def test_error_is_cleared_after_a_valid_answer(graph, context):
    start(graph, context)
    resume(graph, context, {"action": "maybe"})
    payload = pending(resume(graph, context, {"action": "revise", "feedback": "shorter"}))
    assert payload["error"] == ""


def test_no_revisions_left_after_max_drafts(graph, context, writer):
    start(graph, context)
    for _ in range(MAX_DRAFTS - 1):
        resume(graph, context, {"action": "revise", "feedback": "again"})
    payload = pending(resume(graph, context, {"action": "revise", "feedback": "again"}))
    assert len(writer.prompts) == MAX_DRAFTS
    assert payload["revisions_left"] == 0
    assert "No revisions left" in payload["error"]


# --- Threads and persistence -----------------------------------------------

def test_threads_are_independent(graph, context):
    start(graph, context, thread="alice")
    start(graph, context, thread="bob")
    resume(graph, context, {"action": "cancel"}, thread="alice")
    assert graph.get_state(config("alice")).values["status"] == "cancelled"
    assert graph.get_state(config("bob")).next == ("human_review",)


def test_resume_after_restart_with_sqlite(context, tmp_path):
    db = str(tmp_path / "checkpoints.db")

    with SqliteSaver.from_conn_string(db) as saver:  # first "process"
        start(build_graph(saver), context)

    with SqliteSaver.from_conn_string(db) as saver:  # a fresh one, after a "restart"
        graph = build_graph(saver)
        assert graph.get_state(config()).next == ("human_review",)
        result = resume(graph, context, {"action": "approve"})

    assert result["status"] == "sent"
