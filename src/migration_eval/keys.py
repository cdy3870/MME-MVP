"""Provider API keys, read from the environment and nowhere else.

Shared by the CLI's `run` command and the server, so both refuse the same way. A key is never
printed, logged, written to a file or sent to a browser: callers only ever learn WHICH variable
is missing, or whether one is set.

`cli.py` has its own copy of `KEY_VARIABLES` and `_read_provider_keys`; when this is implemented,
make the CLI use it and delete that copy.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

KEY_VARIABLES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}


class MissingKeys(ValueError):
    """One or more providers have no key. The message names the variables, never a value."""

    def __init__(self, variables: list[str]) -> None:
        super().__init__(f"set {', '.join(variables)} in the environment")
        self.variables = variables


def key_status(environ: Mapping[str, str] | None = None) -> dict[str, bool]:
    """Whether a non-blank key is set for each provider in `KEY_VARIABLES`. `environ` defaults to
    `os.environ`. For showing the user which providers they could allow; it returns no values."""
    environ = os.environ if environ is None else environ
    return {provider: bool(environ.get(variable, "").strip()) for provider, variable in KEY_VARIABLES.items()}


def read_keys(providers: list[str], environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The keys for exactly `providers`. Raise `ValueError` for a provider we have no builder
    for, and `MissingKeys` (listing every missing variable at once) when a key is unset or blank.
    A key for a provider that was not asked for is never read into the result.
    """
    environ = os.environ if environ is None else environ

    unknown = sorted(set(providers) - set(KEY_VARIABLES))
    if unknown:
        raise ValueError(f"no support for provider {', '.join(map(repr, unknown))} (supported: {', '.join(sorted(KEY_VARIABLES))})")

    keys: dict[str, str] = {}
    missing: list[str] = []
    for provider in dict.fromkeys(providers):  # each provider once, in the order given
        value = environ.get(KEY_VARIABLES[provider], "").strip()
        if value:
            keys[provider] = value
        else:
            missing.append(KEY_VARIABLES[provider])
    if missing:
        raise MissingKeys(missing)
    return keys
