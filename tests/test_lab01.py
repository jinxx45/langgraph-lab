"""Lab 01 tests. Fake models make these fast and deterministic: they check the
graph's logic (looping, stopping, passing feedback), not how funny the jokes are.
"""

from langchain_core.messages import AIMessage

from labs.lab01_joke_rater.graph import (
    MAX_ATTEMPTS,
    PASS_SCORE,
    Models,
    Rating,
    graph,
    should_retry,
)


class FakeWriter:
    """Returns "joke #1", "joke #2", ... and remembers every prompt it got."""

    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(str(prompt))
        return AIMessage(content=f"joke #{len(self.prompts)}")


class FakeJudge:
    """Hands out the given scores in order."""

    def __init__(self, scores):
        self.scores = list(scores)

    def invoke(self, prompt):
        return Rating(score=self.scores.pop(0), feedback="add a twist ending")


def run(scores):
    writer = FakeWriter()
    result = graph.invoke({"topic": "cats"}, context=Models(writer, FakeJudge(scores)))
    return result, writer


# --- Node tests: one function, no graph ------------------------------------

def test_should_retry_done_when_score_passes():
    assert should_retry({"score": PASS_SCORE, "attempts": 1}) == "done"


def test_should_retry_retries_when_score_low():
    assert should_retry({"score": PASS_SCORE - 1, "attempts": 1}) == "retry"


def test_should_retry_done_when_out_of_attempts():
    assert should_retry({"score": 1, "attempts": MAX_ATTEMPTS}) == "done"


# --- Graph tests: the whole flow, with fake models -------------------------

def test_stops_when_first_joke_is_good():
    result, writer = run([9])
    assert len(writer.prompts) == 1
    assert result["joke"] == "joke #1"
    assert result["score"] == 9
    assert result["attempts"] == 1


def test_retries_until_a_joke_passes():
    result, writer = run([3, 4, 8])
    assert len(writer.prompts) == 3
    assert result["joke"] == "joke #3"
    assert result["score"] == 8


def test_gives_up_after_max_attempts():
    result, writer = run([2] * MAX_ATTEMPTS)
    assert len(writer.prompts) == MAX_ATTEMPTS
    assert result["attempts"] == MAX_ATTEMPTS
    assert result["score"] == 2


def test_prompt_includes_topic():
    _, writer = run([9])
    assert "cats" in writer.prompts[0]


def test_feedback_reaches_the_next_attempt():
    _, writer = run([3, 9])
    assert "add a twist ending" not in writer.prompts[0]
    assert "add a twist ending" in writer.prompts[1]
