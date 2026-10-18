"""Server-sent events decoding, shared by every streaming provider."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_sse(text: str) -> list[Any]:
    """Return the JSON payload of every `data:` event in an SSE body.

    Events are separated by a blank line. `data: [DONE]` (OpenAI's end marker) and
    payloads that are not JSON are skipped. Anthropic's `event:` lines are ignored
    because each payload also carries its own `type`.
    """
    events: list[Any] = []
    for block in re.split(r"\r?\n\r?\n", text):
        data_lines = [
            line[len("data:") :].removeprefix(" ")
            for line in block.splitlines()
            if line.startswith("data:")
        ]
        if not data_lines:
            continue
        data = "\n".join(data_lines)
        if data.strip() == "[DONE]":
            continue
        try:
            events.append(json.loads(data))
        except json.JSONDecodeError:
            continue
    return events
