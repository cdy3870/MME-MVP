"""The `component(name)` context that tags recorded calls."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

DEFAULT_COMPONENT = "default"

# A ContextVar (not a global) so the tag follows asyncio tasks and threads correctly.
_component: ContextVar[str] = ContextVar("migration_eval_component", default=DEFAULT_COMPONENT)


@contextmanager
def component(name: str) -> Iterator[None]:
    """Tag every model call made inside the block with this component name.

        with component("support-triage"):
            client.chat.completions.create(...)
    """
    if not name:
        raise ValueError("component name must not be empty")
    token = _component.set(name)
    try:
        yield
    finally:
        _component.reset(token)


def current_component() -> str:
    return _component.get()
