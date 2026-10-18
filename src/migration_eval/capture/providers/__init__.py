"""Registry of supported providers."""

from __future__ import annotations

from migration_eval.capture.providers.anthropic_messages import PROVIDER as ANTHROPIC
from migration_eval.capture.providers.base import (
    ParsedRequest,
    ParsedResponse,
    Provider,
    Unsupported,
)
from migration_eval.capture.providers.gemini_generate import PROVIDER as GEMINI
from migration_eval.capture.providers.openai_chat import PROVIDER as OPENAI

PROVIDERS: dict[str, Provider] = {p.name: p for p in (OPENAI, ANTHROPIC, GEMINI)}


def detect(url: str) -> Provider | None:
    """Which provider's generate endpoint is this URL, if any?"""
    for provider in PROVIDERS.values():
        if provider.matches(url):
            return provider
    return None


__all__ = ["PROVIDERS", "ParsedRequest", "ParsedResponse", "Provider", "Unsupported", "detect"]
