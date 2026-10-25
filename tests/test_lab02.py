"""Lab 02 tests: does each kind of ticket reach the right node?"""

import pytest
from langchain_core.messages import AIMessage

from labs.lab02_ticket_router.graph import (
    HANDOFF_REPLY,
    INSTRUCTIONS,
    MAX_FORMAT_RETRIES,
    MIN_CONFIDENCE,
    Models,
    Triage,
    classify_prompt,
    graph,
    route_ticket,
)

CATEGORIES = ["billing", "technical", "account", "general"]


class ScriptedClassifier:
    """Plays back one reply per call. Each reply is turned into a Triage, so an
    invalid one (e.g. confidence=500) raises ValidationError, like a real bad answer."""

    def __init__(self, *replies: dict):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return Triage(**self.replies.pop(0))


class FakeWriter:
    def __init__(self):
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(str(messages))
        return AIMessage(content="drafted reply")


def triage(category="billing", urgency="low", confidence=90):
    return {"reasoning": "test", "category": category, "urgency": urgency, "confidence": confidence}


INVALID = triage(confidence=500)


def run(*replies, ticket="My payment failed"):
    classifier, writer = ScriptedClassifier(*replies), FakeWriter()
    result = graph.invoke({"ticket": ticket}, context=Models(classifier, writer))
    return result, writer, classifier


# --- route_ticket on its own ------------------------------------------------

@pytest.mark.parametrize("category", CATEGORIES)
def test_route_confident_low_urgency_to_category(category):
    state = {"category": category, "urgency": "low", "confidence": 90}
    assert route_ticket(state) == category


def test_route_high_urgency_to_human():
    assert route_ticket({"category": "general", "urgency": "high", "confidence": 99}) == "human"


def test_route_low_confidence_to_human():
    state = {"category": "billing", "urgency": "low", "confidence": MIN_CONFIDENCE - 1}
    assert route_ticket(state) == "human"


def test_route_confidence_exactly_at_threshold_is_trusted():
    state = {"category": "billing", "urgency": "low", "confidence": MIN_CONFIDENCE}
    assert route_ticket(state) == "billing"


def test_route_classification_error_to_human():
    assert route_ticket({"classification_error": "bad json"}) == "human"


# --- The prompt ------------------------------------------------------------

def test_prompt_states_the_rules_the_schema_cannot():
    prompt = classify_prompt("hello")
    assert "locked out" in prompt  # urgency rule
    assert "0 to 100" in prompt    # confidence range
    assert "<ticket>\nhello\n</ticket>" in prompt


# --- The whole graph -------------------------------------------------------

@pytest.mark.parametrize("category", CATEGORIES)
def test_specialist_drafts_reply_with_its_own_instructions(category):
    result, writer, _ = run(triage(category=category))
    assert result["route"] == category
    assert result["reply"] == "drafted reply"
    assert len(writer.prompts) == 1
    assert INSTRUCTIONS[category] in writer.prompts[0]


def test_ticket_text_reaches_the_specialist():
    _, writer, _ = run(triage(), ticket="I was charged twice")
    assert "I was charged twice" in writer.prompts[0]


def test_urgent_ticket_goes_to_human_without_calling_llm():
    result, writer, _ = run(triage(urgency="high"))
    assert result["route"] == "human"
    assert result["reply"] == HANDOFF_REPLY
    assert writer.prompts == []


def test_classification_is_kept_in_state():
    result, _, _ = run(triage(category="account", urgency="low", confidence=75))
    assert result["category"] == "account"
    assert result["urgency"] == "low"
    assert result["confidence"] == 75


# --- Bad answers: retry with the error, then fall back ----------------------

def test_invalid_answer_is_retried_with_the_error():
    result, _, classifier = run(INVALID, triage(category="technical"))
    assert result["route"] == "technical"
    assert len(classifier.prompts) == 2
    assert "previous answer was invalid" not in classifier.prompts[0]
    assert "previous answer was invalid" in classifier.prompts[1]
    assert "confidence" in classifier.prompts[1]  # the error names the bad field
    assert "classification_error" not in result


def test_hands_off_to_human_after_max_retries():
    result, writer, classifier = run(*[INVALID] * (MAX_FORMAT_RETRIES + 1))
    assert len(classifier.prompts) == MAX_FORMAT_RETRIES + 1
    assert result["route"] == "human"
    assert result["reply"] == HANDOFF_REPLY
    assert "confidence" in result["classification_error"]
    assert writer.prompts == []
