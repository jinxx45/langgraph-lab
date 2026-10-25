"""Lab 02 — Support Ticket Router: classify → route to a specialist or a human.

Run it:  uv run python -m labs.lab02_ticket_router.graph "I was charged twice"
"""

from dataclasses import dataclass
from typing import Literal, TypedDict

from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy
from pydantic import BaseModel, Field, ValidationError

MODEL = "qwen2.5:7b"
MIN_CONFIDENCE = 60
MAX_FORMAT_RETRIES = 2

Category = Literal["billing", "technical", "account", "general"]

CATEGORY_GUIDE = """\
- billing: payments, refunds, double charges, invoices, subscription price
- technical: crashes, bugs, error messages, slowness, something not working
- account: login, password, OTP, profile changes, KYC, closing the account
- general: questions about features, feedback, anything else"""

URGENCY_GUIDE = """\
- high: the customer lost money, is locked out of their account, or a core feature is down
- low: everything else"""

INSTRUCTIONS: dict[str, str] = {
    "billing": (
        "You are a billing specialist. Acknowledge the issue, never promise a "
        "refund amount, and say the team will verify the transaction within "
        "2 business days."
    ),
    "technical": (
        "You are a technical support engineer. Ask for the device, app version "
        "and exact error message, and suggest one safe troubleshooting step."
    ),
    "account": (
        "You are an account security specialist. Never ask for passwords or "
        "OTPs. Point the customer to the secure reset or settings flow."
    ),
    "general": "You are a friendly support agent. Answer briefly and helpfully.",
}

HANDOFF_REPLY = (
    "Thanks for reaching out. Your ticket has been passed to a member of our "
    "team, who will get back to you shortly."
)


# --- State -----------------------------------------------------------------

class TicketState(TypedDict, total=False):
    ticket: str
    category: Category
    urgency: Literal["low", "high"]
    confidence: int
    route: str  # which node handled it: a category name or "human"
    reply: str
    classification_error: str  # set only if every classify attempt was invalid


# --- Structured output -----------------------------------------------------

class Triage(BaseModel):
    # Generated first, so the model explains before it commits to a category.
    reasoning: str
    category: Category
    urgency: Literal["low", "high"]
    # 0-100, not 0-1: small models reply "100" for a 0-1 field, which fails validation.
    confidence: int = Field(ge=0, le=100)


# --- Context ---------------------------------------------------------------

@dataclass
class Models:
    classifier: Runnable  # .invoke(prompt) -> Triage
    writer: Runnable      # .invoke(messages) -> message with .content


# --- Nodes -----------------------------------------------------------------

def classify_prompt(ticket: str) -> str:
    # The schema only enforces the shape; the meaning of each field has to be
    # in the prompt, because Ollama never shows Field descriptions to the model.
    return (
        "You triage customer support tickets.\n\n"
        f"Categories:\n{CATEGORY_GUIDE}\n\n"
        f"Urgency:\n{URGENCY_GUIDE}\n\n"
        "Confidence: a whole number from 0 to 100 for how sure you are about the category.\n\n"
        "Answer with: reasoning (one sentence), category, urgency, confidence.\n\n"
        f"<ticket>\n{ticket}\n</ticket>"
    )


def classify(state: TicketState, runtime: Runtime[Models]) -> dict:
    prompt = classify_prompt(state["ticket"])
    for _ in range(MAX_FORMAT_RETRIES + 1):
        try:
            triage: Triage = runtime.context.classifier.invoke(prompt)
            return {
                "category": triage.category,
                "urgency": triage.urgency,
                "confidence": triage.confidence,
            }
        except (ValidationError, OutputParserException) as err:
            error = str(err)[:500]
            # Show the model its mistake so the next attempt can fix it.
            prompt += f"\n\nYour previous answer was invalid:\n{error}\nFix it and answer again."
    return {"classification_error": error}


def route_ticket(state: TicketState) -> str:
    # Business rules, not an LLM: failed, urgent or unclear tickets go to a person.
    if state.get("classification_error"):
        return "human"
    if state["urgency"] == "high" or state["confidence"] < MIN_CONFIDENCE:
        return "human"
    return state["category"]


def _draft_reply(category: str, state: TicketState, runtime: Runtime[Models]) -> dict:
    messages = [("system", INSTRUCTIONS[category]), ("human", state["ticket"])]
    reply = runtime.context.writer.invoke(messages)
    return {"route": category, "reply": reply.content.strip()}


def billing_agent(state: TicketState, runtime: Runtime[Models]) -> dict:
    return _draft_reply("billing", state, runtime)


def technical_agent(state: TicketState, runtime: Runtime[Models]) -> dict:
    return _draft_reply("technical", state, runtime)


def account_agent(state: TicketState, runtime: Runtime[Models]) -> dict:
    return _draft_reply("account", state, runtime)


def general_agent(state: TicketState, runtime: Runtime[Models]) -> dict:
    return _draft_reply("general", state, runtime)


def human_handoff(state: TicketState) -> dict:
    return {"route": "human", "reply": HANDOFF_REPLY}


# --- Wiring ----------------------------------------------------------------

builder = StateGraph(TicketState, context_schema=Models)
# RetryPolicy re-runs the node on infrastructure errors (Ollama down, timeouts).
# Bad answers are a ValueError, which it skips: classify() handles those itself.
builder.add_node("classify", classify, retry_policy=RetryPolicy(max_attempts=3))
builder.add_node("billing_agent", billing_agent)
builder.add_node("technical_agent", technical_agent)
builder.add_node("account_agent", account_agent)
builder.add_node("general_agent", general_agent)
builder.add_node("human_handoff", human_handoff)

builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    route_ticket,
    {
        "billing": "billing_agent",
        "technical": "technical_agent",
        "account": "account_agent",
        "general": "general_agent",
        "human": "human_handoff",
    },
)
for node in ["billing_agent", "technical_agent", "account_agent", "general_agent", "human_handoff"]:
    builder.add_edge(node, END)

graph = builder.compile()


def default_models() -> Models:
    from langchain_ollama import ChatOllama

    return Models(
        classifier=ChatOllama(model=MODEL, temperature=0).with_structured_output(Triage),
        writer=ChatOllama(model=MODEL, temperature=0.3),
    )


if __name__ == "__main__":
    import sys

    ticket = " ".join(sys.argv[1:]) or "I was charged twice for my monthly plan."
    for update in graph.stream({"ticket": ticket}, context=default_models(), stream_mode="updates"):
        print(update, "\n")
