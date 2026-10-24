"""Lab 01 evals: run the real graph over a small dataset and measure quality.

Run it:  uv run python -m evals.lab01_eval
"""

import json
import time
from datetime import datetime
from pathlib import Path

from labs.lab01_joke_rater.graph import PASS_SCORE, default_models, graph, judge_prompt

TOPICS = [
    "cats",
    "programmers",
    "Monday mornings",
    "Bangalore traffic",
    "exams",
    "coffee",
    "WiFi",
    "cricket",
]

RESULTS_DIR = Path(__file__).parent / "results"


def judge_consistency(judge, joke, topic, runs=3):
    """Rate the same joke several times. A trustworthy judge gives the same score."""
    prompt = judge_prompt(topic, joke)
    return [judge.invoke(prompt).score for _ in range(runs)]


def main():
    models = default_models()

    rows = []
    for topic in TOPICS:
        start = time.perf_counter()
        result = graph.invoke({"topic": topic}, context=models)
        rows.append(
            {
                "topic": topic,
                "score": result["score"],
                "attempts": result["attempts"],
                "passed": result["score"] >= PASS_SCORE,
                "seconds": round(time.perf_counter() - start, 1),
                "joke": result["joke"],
            }
        )
        r = rows[-1]
        print(f"{topic:<20} score={r['score']:<3} attempts={r['attempts']}  {r['seconds']}s")

    n = len(rows)
    summary = {
        "pass_rate": sum(r["passed"] for r in rows) / n,
        "first_try_pass_rate": sum(r["passed"] and r["attempts"] == 1 for r in rows) / n,
        "avg_score": sum(r["score"] for r in rows) / n,
        "avg_attempts": sum(r["attempts"] for r in rows) / n,
        "avg_seconds": sum(r["seconds"] for r in rows) / n,
        "judge_consistency": judge_consistency(models.judge, rows[0]["joke"], rows[0]["topic"]),
    }

    print("\n--- summary ---")
    for key, value in summary.items():
        print(f"{key:<20} {value if isinstance(value, list) else round(value, 2)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"lab01-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
