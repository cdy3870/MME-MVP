"""Write a dummy v0 records file (JSONL) for smoke-testing a run end to end.

Every record is a replay-grade support-ticket classification call: a system prompt asking for
one word (billing, bug, account or other), a short ticket, temperature 0 and a 16-token limit,
with the logged answer. A one-word answer suits the exact-text comparator used until the judge
(EVL-5) exists, and keeps a smoke run's provider spend to cents.

The tickets are made up and the logged answers are what the ticket was written to be, not real
model output. Generation is seeded, so the same options give the same file.

    python scripts/make_dummy_records.py                       # 60 records -> records.dummy.jsonl
    python scripts/make_dummy_records.py -n 120 -o out.jsonl --provider openai --model gpt-4o-mini

Runs on any Python 3.9+, with no dependencies. The model must be in catalog/catalog.yaml for the
baseline replays to be priced and metered.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

SYSTEM_PROMPT = (
    "You triage customer support tickets. Classify the ticket as exactly one of: billing, bug, "
    "account, other. Reply with that one word in lowercase and nothing else."
)

# (label, ticket templates); {thing}, {when} and {detail} are filled from the lists below.
TICKETS: dict[str, list[str]] = {
    "billing": [
        "I was charged twice for my {thing} {when}.",
        "Why did my invoice go up? I'm still on the same {thing}.",
        "Please refund the last payment for my {thing}, I cancelled {when}.",
        "My card was declined but the {thing} still shows as active. Am I being billed?",
        "Can I get a receipt with my company's VAT number for the {thing}?",
    ],
    "bug": [
        "The export button does nothing since {when}. {detail}",
        "The app crashes whenever I open the {thing} page. {detail}",
        "Search returns no results even for exact names. {detail}",
        "Notifications arrive twice since the update {when}. {detail}",
        "Charts on the dashboard are blank. {detail}",
    ],
    "account": [
        "I can't log in, the password reset email never arrives.",
        "How do I change the email address on my account?",
        "Please delete my account and all my data.",
        "I need to transfer ownership of the workspace to a colleague.",
        "Two-factor codes stopped working after I got a new phone {when}.",
    ],
    "other": [
        "Do you have an office in Berlin?",
        "Is there a public roadmap I can follow?",
        "I'd like to suggest a dark mode for the {thing} page.",
        "Are you hiring for support roles?",
        "What's the difference between your plans for a team of five?",
    ],
}
THINGS = ["subscription", "Pro plan", "team plan", "annual plan", "reports", "settings", "billing"]
WHENS = ["this month", "yesterday", "last week", "on Monday", "two days ago", "this morning"]
DETAILS = [
    "I'm on Chrome on a Mac.",
    "Happens on both Windows and Linux.",
    "Clearing the cache didn't help.",
    "Other people on my team see it too.",
    "It worked fine before.",
    "",
]


def make_records(count: int, component: str, provider: str, model: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    labels = list(TICKETS)
    clock = datetime(2026, 11, 2, 9, 0, tzinfo=timezone.utc)
    records = []
    for i in range(1, count + 1):
        label = labels[(i - 1) % len(labels)]  # an even mix of the four classes
        ticket = rng.choice(TICKETS[label]).format(
            thing=rng.choice(THINGS), when=rng.choice(WHENS), detail=rng.choice(DETAILS)
        ).strip()
        clock += timedelta(seconds=rng.randint(20, 240))  # spread over a working day
        latency_ms = rng.randint(350, 1400)
        records.append(
            {
                "schema_version": "v0",
                "id": f"rec_{i:04d}",
                "component": component,
                "provider": provider,
                "model": model,
                "started_at": _iso(clock),
                "completed_at": _iso(clock + timedelta(milliseconds=latency_ms)),
                "latency": {"total_ms": latency_ms},
                "usage": {"input_tokens": 60 + len(ticket) // 4, "output_tokens": 2},
                "request": {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": ticket},
                    ],
                    "parameters": {"temperature": 0, "max_output_tokens": 16},
                    "stream": False,
                },
                "response": {
                    "status": "success",
                    "id": f"msg_dummy_{i:04d}",
                    "model": model,
                    "text": label,
                    "finish_reason": "stop",
                },
                "metadata": {"category": "classification", "dummy": True},
                "source": {"type": "recorder", "tool_version": "0.1.0"},
            }
        )
    return records


def _iso(t: datetime) -> str:
    return t.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-n", "--count", type=int, default=60, help="records to write (smoke runs sample 50)")
    p.add_argument("-o", "--out", type=Path, default=Path("records.dummy.jsonl"))
    p.add_argument("--component", default="support-triage")
    p.add_argument("--provider", default="anthropic")
    p.add_argument("--model", default="claude-haiku-5-5", help="the 'current model'; must be in the catalog")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    if args.count < 1:
        p.error("--count must be at least 1")

    records = make_records(args.count, args.component, args.provider, args.model, args.seed)
    with args.out.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} records for component {args.component!r} ({args.provider}/{args.model}) to {args.out}")


if __name__ == "__main__":
    main()
