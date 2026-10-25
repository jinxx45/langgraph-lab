"""Lab 02 evals: measure the classifier against human labels.

Run it:  uv run python -m evals.lab02_eval
"""

import json
from datetime import datetime
from pathlib import Path

from labs.lab02_ticket_router.graph import (
    MIN_CONFIDENCE,
    classify_prompt,
    default_models,
    route_ticket,
)

CATEGORIES = ["billing", "technical", "account", "general"]
DATA = Path(__file__).parent / "data" / "lab02_tickets.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


def load_dataset():
    return [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]


def precision_recall(rows, category):
    predicted = [r for r in rows if r["predicted"] == category]
    actual = [r for r in rows if r["category"] == category]
    hits = [r for r in predicted if r["category"] == category]
    precision = len(hits) / len(predicted) if predicted else 0.0
    recall = len(hits) / len(actual) if actual else 0.0
    return precision, recall


def print_confusion_matrix(rows):
    print(f"\n{'actual ↓ / predicted →':<24}" + "".join(f"{c[:9]:>11}" for c in CATEGORIES))
    for actual in CATEGORIES:
        counts = [
            sum(r["category"] == actual and r["predicted"] == pred for r in rows) for pred in CATEGORIES
        ]
        print(f"{actual:<24}" + "".join(f"{n:>11}" for n in counts))


def accuracy(rows):
    return sum(r["correct"] for r in rows) / len(rows) if rows else float("nan")


def main():
    # Eval the classifier alone: routing accuracy doesn't depend on the replies.
    classifier = default_models().classifier

    rows = []
    for item in load_dataset():
        t = classifier.invoke(classify_prompt(item["ticket"]))
        route = route_ticket({"category": t.category, "urgency": t.urgency, "confidence": t.confidence})
        rows.append(
            {
                **item,
                "predicted": t.category,
                "predicted_urgency": t.urgency,
                "confidence": t.confidence,
                "route": route,
                "correct": t.category == item["category"],
                "reasoning": t.reasoning,
            }
        )
        r = rows[-1]
        mark = "✓" if r["correct"] else "✗"
        print(f"{mark} #{r['id']:<3} {r['category']:<10}→ {r['predicted']:<10} conf={r['confidence']:<3}  route={route}")

    print_confusion_matrix(rows)

    print("\nper category:")
    for c in CATEGORIES:
        p, rc = precision_recall(rows, c)
        print(f"  {c:<10} precision={p:.2f}  recall={rc:.2f}")

    urgent = [r for r in rows if r["urgency"] == "high"]
    confident = [r for r in rows if r["confidence"] >= MIN_CONFIDENCE]
    unsure = [r for r in rows if r["confidence"] < MIN_CONFIDENCE]

    summary = {
        "accuracy": accuracy(rows),
        "urgency_accuracy": sum(r["predicted_urgency"] == r["urgency"] for r in rows) / len(rows),
        "escalation_rate": sum(r["route"] == "human" for r in rows) / len(rows),
        # The dangerous error: an urgent ticket answered by a bot instead of a person.
        "missed_urgent": sum(r["route"] != "human" for r in urgent),
        "accuracy_when_confident": accuracy(confident),
        "accuracy_when_unsure": accuracy(unsure),
        "n_unsure": len(unsure),
    }

    print("\n--- summary ---")
    for key, value in summary.items():
        print(f"{key:<26} {round(value, 2) if isinstance(value, float) else value}")

    wrong = [r for r in rows if not r["correct"]]
    if wrong:
        print("\nmistakes:")
        for r in wrong:
            print(f"  #{r['id']} expected {r['category']}, got {r['predicted']}: {r['ticket']}")
            print(f"      model said: {r['reasoning']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"lab02-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
