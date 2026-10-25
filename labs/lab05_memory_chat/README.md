# Lab 05 — Chat That Remembers You

**Goal:** a chat assistant that remembers facts about you — your name,
your job, what you like — not just within one conversation, but in **new
conversations, days later**.

```
START → remember → respond → END
           │          │
           ▼          ▼
     ┌────────── Store ──────────┐
     │ ("memories", "pragadeesh")│   long-term: survives every conversation
     │   m1: "Name is Pragadeesh"│
     │   m2: "Learning LangGraph"│
     └───────────────────────────┘
   + checkpointer: this thread's messages (short-term)
```

## Two kinds of memory

| | Short-term | Long-term |
|---|---|---|
| What | The messages of *this* conversation | Facts about the *user* |
| Stored by | Checkpointer (Lab 04) | **Store** (new) |
| Keyed by | `thread_id` | `user_id` (a namespace) |
| New conversation | Starts empty | Still there |
| Example | "As I said above..." | "You told me last week you're a Java dev" |

A thread is one chat. A user has many threads. The checkpointer can't
answer "what do I know about this person?" across threads — the Store can.

## New ideas

**1. The Store — a key-value database for facts.**

```python
store.put(("memories", user_id), "a1b2", {"text": "Works as a Java developer"})
store.search(("memories", user_id))   # every memory for this user
store.delete(("memories", user_id), "a1b2")
```

The **namespace** `("memories", user_id)` is like a folder path: each
user's memories live in their own folder, so users never see each other's.
Nodes get the store through `runtime.store`, the same way they get
`runtime.context`.

**2. Memory is written by an LLM, checked by code.** After every message,
`remember` asks the model: *"Is there a lasting fact here? Did the user ask
you to forget something?"* It returns a structured `MemoryUpdate`:

```python
class MemoryUpdate(BaseModel):
    remove: list[str]   # known facts to delete, copied as text
    add: list[str]      # new facts
```

The model *proposes*; our code *applies* — removing only facts that really
exist, skipping duplicates and anything that looks like a secret, and
enforcing a size limit.

**What went wrong first** (worth knowing — these are real findings):

1. `remove` was a list of **ids** like `"f8960390"`. qwen2.5 returned whole
   lines (`"[f8960390] City: Chennai"`) and paired ids with the **wrong**
   fact. The "ignore unknown ids" check prevented any damage, but forgetting
   didn't work. Fix: remove by **text** — small models copy text well and
   random codes badly. `normalize()` forgives small differences.
2. "I moved to Pune" removed Chennai but never added Pune. Fix: put
   `remove` **before** `add`. Fields are generated in order; once the model
   has written down what changed, it writes the replacement.
3. The chat said *"Sure, I'll update my information"* while nothing had
   changed. **The reply and the memory are separate systems** — what the bot
   says is not proof of what was stored. That's why `/memories` exists.

**3. Memory goes into the system prompt.** `respond` loads the user's
memories and puts them in front of the conversation, so the model "knows"
you without the facts ever being part of the chat history.

**4. Trimming the history.** Every message is sent on every turn. Long
chats get slow and eventually overflow the model's context window. We send
only the last `MAX_HISTORY` messages — the important facts survive anyway,
because they're in the Store.

## Production details in this lab

| Problem | What we do |
|---|---|
| User A sees user B's memories | Namespaces per `user_id`; it comes from the context, never from the chat |
| The model "forgets" something that isn't stored | Only facts that actually exist are removed |
| The model stores an OTP or card number anyway | `looks_sensitive()` blocks long digit runs in code |
| Memory grows forever | `MAX_MEMORIES`; new facts beyond it are dropped |
| Same fact saved ten times | Exact duplicates (ignoring case) are skipped |
| "Please forget my address" | Supported: privacy is a feature, not an afterthought |
| Chat history overflows the model | Only the last `MAX_HISTORY` messages are sent |
| Memory extraction fails | It's skipped for that turn; the chat still answers |

## Run it

```bash
uv run pytest tests/test_lab05.py -v
uv run python -m labs.lab05_memory_chat.graph --user pragadeesh
```

Tell it something about yourself, type `/new` to start a fresh
conversation, and ask what it knows. Commands: `/memories`, `/new`, `/quit`.

## Think about it

- What should **not** be remembered? (Passwords, card numbers, health
  details told in passing...)
- The model decides what's a "lasting fact". What if it saves something
  wrong — how would the user find out and fix it?
- Search returns *all* memories here. With 5,000 memories you'd need
  semantic search (embeddings). The Store supports that with an `index`.
