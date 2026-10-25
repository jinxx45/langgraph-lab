"""Lab 05 tests: memory within a thread, across threads, and between users."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from labs.lab05_memory_chat.graph import (
    MAX_HISTORY,
    MAX_MEMORIES,
    Context,
    MemoryUpdate,
    build_graph,
    load_memories,
    looks_sensitive,
)


class FakeChat:
    """Replies "ok" and records the messages of every call."""

    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage("ok")

    def last_system_prompt(self):
        return self.calls[-1][0].content


class ScriptedExtractor:
    """Plays back one MemoryUpdate per call (dicts are validated like real output);
    once the script runs out, it reports no changes."""

    def __init__(self, *updates):
        self.updates = list(updates)

    def invoke(self, prompt):
        return MemoryUpdate.model_validate(self.updates.pop(0) if self.updates else {})


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def graph(store):
    return build_graph(InMemorySaver(), store)


def say(graph, text, extractor=None, chat=None, user="asha", thread="t1"):
    chat = chat or FakeChat()
    context = Context(user_id=user, chat=chat, extractor=extractor or ScriptedExtractor())
    result = graph.invoke({"messages": [("human", text)]}, {"configurable": {"thread_id": thread}}, context=context)
    return result, chat


def texts(store, user="asha"):
    return sorted(load_memories(store, user).values())


# --- Saving memories -------------------------------------------------------

def test_new_fact_is_saved(graph, store):
    say(graph, "I'm a Java developer", ScriptedExtractor({"add": ["Works as a Java developer"]}))
    assert texts(store) == ["Works as a Java developer"]


def test_saved_fact_is_used_in_the_same_turn(graph):
    _, chat = say(graph, "My name is Asha", ScriptedExtractor({"add": ["Name is Asha"]}))
    assert "Name is Asha" in chat.last_system_prompt()


def test_duplicates_are_skipped_ignoring_case(graph, store):
    say(graph, "a", ScriptedExtractor({"add": ["Likes cricket"]}))
    say(graph, "b", ScriptedExtractor({"add": ["likes CRICKET", "Likes cricket"]}))
    assert texts(store) == ["Likes cricket"]


@pytest.mark.parametrize("fact", ["OTP is 482913", "Card 4111 1111 1111 1111", "Phone is 98765-43210"])
def test_sensitive_numbers_are_never_stored(graph, store, fact):
    say(graph, "x", ScriptedExtractor({"add": [fact]}))
    assert texts(store) == []


def test_ordinary_numbers_are_fine():
    assert not looks_sensitive("Born in 1998")
    assert not looks_sensitive("Has 2 cats")


def test_memory_limit(graph, store):
    say(graph, "x", ScriptedExtractor({"add": [f"Fact {i}" for i in range(MAX_MEMORIES + 5)]}))
    assert len(texts(store)) == MAX_MEMORIES


def test_failed_extraction_does_not_break_the_chat(graph, store):
    result, _ = say(graph, "hello", ScriptedExtractor({"add": "not a list"}))
    assert result["messages"][-1].content == "ok"
    assert texts(store) == []


# --- Removing memories -----------------------------------------------------

def test_forget_removes_the_memory(graph, store):
    say(graph, "a", ScriptedExtractor({"add": ["Lives in Chennai", "Likes tea"]}))
    say(graph, "forget where I live", ScriptedExtractor({"remove": ["Lives in Chennai"]}))
    assert texts(store) == ["Likes tea"]


@pytest.mark.parametrize("copied", ["lives in chennai", "- Lives in Chennai", "Lives in Chennai."])
def test_forget_tolerates_small_copy_differences(graph, store, copied):
    say(graph, "a", ScriptedExtractor({"add": ["Lives in Chennai"]}))
    say(graph, "forget it", ScriptedExtractor({"remove": [copied]}))
    assert texts(store) == []


def test_correction_replaces_the_old_fact(graph, store):
    say(graph, "a", ScriptedExtractor({"add": ["Lives in Chennai"]}))
    say(graph, "I moved to Pune", ScriptedExtractor({"remove": ["Lives in Chennai"], "add": ["Lives in Pune"]}))
    assert texts(store) == ["Lives in Pune"]


def test_removing_a_fact_that_does_not_exist_changes_nothing(graph, store):
    say(graph, "a", ScriptedExtractor({"add": ["Likes tea"]}))
    say(graph, "b", ScriptedExtractor({"remove": ["Likes coffee"]}))
    assert texts(store) == ["Likes tea"]


# --- Short-term vs long-term -----------------------------------------------

def test_same_thread_remembers_the_conversation(graph):
    say(graph, "first message")
    _, chat = say(graph, "second message")
    sent = [m.content for m in chat.calls[-1] if isinstance(m, HumanMessage)]
    assert sent == ["first message", "second message"]


def test_new_thread_has_empty_history_but_keeps_memories(graph):
    say(graph, "I'm Asha", ScriptedExtractor({"add": ["Name is Asha"]}), thread="monday")
    _, chat = say(graph, "who am I?", thread="friday")
    sent = [m.content for m in chat.calls[-1] if isinstance(m, HumanMessage)]
    assert sent == ["who am I?"]  # Monday's chat is not in Friday's history...
    assert "Name is Asha" in chat.last_system_prompt()  # ...but the fact is remembered


def test_users_never_see_each_others_memories(graph, store):
    say(graph, "x", ScriptedExtractor({"add": ["Name is Asha"]}), user="asha")
    _, chat = say(graph, "who am I?", user="ravi")
    assert "Asha" not in chat.last_system_prompt()
    assert texts(store, "ravi") == []


def test_only_recent_history_is_sent(graph):
    chat = FakeChat()
    for i in range(MAX_HISTORY):
        say(graph, f"message {i}", chat=chat)
    sent = chat.calls[-1]
    assert isinstance(sent[0], SystemMessage)
    assert len(sent) == MAX_HISTORY + 1  # system prompt + the most recent messages
    assert sent[-1].content == f"message {MAX_HISTORY - 1}"
