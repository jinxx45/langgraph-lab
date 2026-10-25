"""Lab 05 — Chat That Remembers You: short-term (thread) + long-term (Store) memory.

Run it:  uv run python -m labs.lab05_memory_chat.graph --user pragadeesh
"""

import re
import uuid
from dataclasses import dataclass

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy
from pydantic import BaseModel, ValidationError

MODEL = "qwen2.5:7b"
MAX_HISTORY = 20   # messages of this conversation sent to the model each turn
MAX_MEMORIES = 50  # lasting facts kept per user


# --- Context ---------------------------------------------------------------

@dataclass
class Context:
    user_id: str             # whose memories to use: set by the app, never by the chat
    chat: BaseChatModel      # .invoke(messages) -> AIMessage
    extractor: Runnable      # .invoke(prompt) -> MemoryUpdate


# --- Structured output -----------------------------------------------------

class MemoryUpdate(BaseModel):
    # Known facts to delete, copied as text. Not ids: small models copy text
    # well but garble random ids like "f8960390" and pair them with the wrong fact.
    remove: list[str] = []
    # Generated after remove: once the model has dropped "City: Chennai", it
    # reliably writes the replacement. In the other order it often forgot.
    add: list[str] = []     # new lasting facts, one short sentence each


# --- Store helpers -----------------------------------------------------------

def namespace(user_id: str) -> tuple[str, ...]:
    # Like a folder path: each user's memories live in their own folder.
    return ("memories", user_id)


def load_memories(store: BaseStore, user_id: str) -> dict[str, str]:
    # search() returns 10 items unless told otherwise.
    items = store.search(namespace(user_id), limit=MAX_MEMORIES)
    return {item.key: item.value["text"] for item in items}


def looks_sensitive(text: str) -> bool:
    # The prompt asks the model not to store secrets; this makes sure.
    # Six or more digits in a row: OTPs, card, account and phone numbers.
    return re.search(r"\d[\d -]{4,}\d", text) is not None


# --- Nodes -----------------------------------------------------------------

def extract_prompt(message: str, memories: dict[str, str]) -> str:
    known = "\n".join(f"- {text}" for text in memories.values()) or "(none yet)"
    return (
        "You keep a list of lasting facts about a user: name, job, city, preferences, "
        "goals, important people.\n"
        "Read the user's latest message and decide:\n"
        "- add: new lasting facts, each one short sentence about the user, "
        'like "Works as a Java developer". Skip small talk, questions and temporary things.\n'
        "- remove: known facts the user asks you to forget, copied exactly as listed.\n"
        "If the message changes a known fact (a new city, a new job), remove the old "
        "fact AND add the new one.\n"
        "Never store passwords, OTPs, card or bank numbers.\n"
        "If nothing changes, return empty lists.\n\n"
        f"Known facts:\n{known}\n\n"
        f"<message>\n{message}\n</message>"
    )


def normalize(text: str) -> str:
    # Match "- City: Chennai", "city: chennai." and "City: Chennai" as the same fact.
    return text.strip().lstrip("-• ").rstrip(".").strip().casefold()


def remember(state: MessagesState, runtime: Runtime[Context]) -> dict:
    last = state["messages"][-1]
    if not isinstance(last, HumanMessage):
        return {}

    store, user_id = runtime.store, runtime.context.user_id
    memories = load_memories(store, user_id)
    try:
        update: MemoryUpdate = runtime.context.extractor.invoke(extract_prompt(last.content, memories))
    except (ValidationError, OutputParserException):
        return {}  # memory is best-effort: a failed extraction must not break the chat

    # The model proposes; the code decides what actually changes.
    to_remove = {normalize(text) for text in update.remove}
    for key, text in list(memories.items()):
        if normalize(text) in to_remove:  # only facts that really exist
            store.delete(namespace(user_id), key)
            del memories[key]

    known = {normalize(text) for text in memories.values()}
    for text in (t.strip() for t in update.add):
        if not text or normalize(text) in known or looks_sensitive(text):
            continue
        if len(memories) >= MAX_MEMORIES:
            break
        key = uuid.uuid4().hex[:8]
        store.put(namespace(user_id), key, {"text": text})
        memories[key] = text
        known.add(normalize(text))
    return {}


def system_prompt(memories: dict[str, str]) -> str:
    facts = "\n".join(f"- {text}" for text in memories.values()) or "- (nothing yet)"
    return (
        "You are a friendly, concise assistant. What you know about the user from "
        f"earlier conversations:\n{facts}\n"
        "Use these facts naturally when relevant. Never claim to know anything that is "
        "not listed or said in this conversation."
    )


def respond(state: MessagesState, runtime: Runtime[Context]) -> dict:
    memories = load_memories(runtime.store, runtime.context.user_id)
    # Only the recent messages: long chats would overflow the model's context.
    history = state["messages"][-MAX_HISTORY:]
    reply = runtime.context.chat.invoke([SystemMessage(system_prompt(memories)), *history])
    return {"messages": [reply]}


# --- Wiring ----------------------------------------------------------------

builder = StateGraph(MessagesState, context_schema=Context)
# remember runs first, so "My name is X. What's my name?" works in one turn.
# Its store writes happen only after the LLM call, so a retry can't double-write.
builder.add_node("remember", remember, retry_policy=RetryPolicy(max_attempts=3))
builder.add_node("respond", respond, retry_policy=RetryPolicy(max_attempts=3))
builder.add_edge(START, "remember")
builder.add_edge("remember", "respond")
builder.add_edge("respond", END)


def build_graph(checkpointer: BaseCheckpointSaver, store: BaseStore):
    # checkpointer = this conversation's messages; store = facts across conversations.
    return builder.compile(checkpointer=checkpointer, store=store)


def default_context(user_id: str) -> Context:
    from langchain_ollama import ChatOllama

    return Context(
        user_id=user_id,
        chat=ChatOllama(model=MODEL, temperature=0.7),
        extractor=ChatOllama(model=MODEL, temperature=0).with_structured_output(MemoryUpdate),
    )


# --- Command line ------------------------------------------------------------

def main() -> None:
    import argparse
    from pathlib import Path

    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.store.sqlite import SqliteStore

    parser = argparse.ArgumentParser(description="Chat with an assistant that remembers you.")
    parser.add_argument("--user", required=True, help="whose memories to use")
    parser.add_argument("--thread", help="continue an earlier conversation")
    args = parser.parse_args()

    data = Path(__file__).resolve().parents[2] / "data"
    data.mkdir(exist_ok=True)
    new_thread = lambda: uuid.uuid4().hex[:8]  # noqa: E731
    thread = args.thread or new_thread()
    context = default_context(args.user)

    with (
        SqliteSaver.from_conn_string(str(data / "lab05_checkpoints.db")) as saver,
        SqliteStore.from_conn_string(str(data / "lab05_memories.db")) as store,
    ):
        store.setup()  # creates the store's tables the first time
        graph = build_graph(saver, store)
        print(f"Chatting as {args.user}, conversation {thread}. Commands: /memories /new /quit")

        while True:
            try:
                text = input("\nyou > ").lstrip("﻿").strip()
            except EOFError:
                break
            if not text:
                continue
            if text == "/quit":
                break
            if text == "/new":
                thread = new_thread()
                print(f"(new conversation {thread}: chat history is empty, memories stay)")
                continue
            if text == "/memories":
                memories = load_memories(store, args.user)
                print("\n".join(f"  [{k}] {t}" for k, t in memories.items()) or "  (none)")
                continue

            config = {"configurable": {"thread_id": thread}}
            result = graph.invoke({"messages": [("human", text)]}, config, context=context)
            print(f"bot > {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
