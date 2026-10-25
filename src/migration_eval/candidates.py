"""Stage 1 of candidate selection (CAT-3): drop offerings that fail on hard facts.

An offering is dropped, with every reason it fails, if any of these hold:

    incumbent        it is the model the traffic already runs on (replaying it is the
                     noise floor, EVL-1, not a candidate)
    context          its context window is smaller than the largest call needs
    tools            the traffic uses tools and the offering does not support them
    region           a region is required and the offering cannot serve it
    hosting option   the offering is not hosted the way the partner asked (MVP: API only)
    allow-list       its provider is not on the partner's allow-list

All reasons are collected, not just the first, because CAT-5 needs each dropped candidate
to keep its reason and a partner wants to see everything that is wrong with it. A reason is
a plain sentence; each check returns it, or None when the offering passes.

Open decisions (see the notes on CAT-3):
    * where the required region comes from; the ERD lists region as a filter but not who sets it
    * the context check has no safety margin, though tokenizers differ by provider
    * when the profile has no token counts the context check cannot run, so it keeps the offering
    * the incumbent is not in the ERD's list of rules, but excluding it is assumed

Stage 2 (predicted cost, CAT-4) is separate and runs on what Stage 1 keeps. The CLI
`migration-eval candidates` and the `POST /candidates` endpoint come later, once uploads exist.
"""

from __future__ import annotations

from dataclasses import dataclass

from migration_eval.catalog import Catalog, Offering
from migration_eval.profile import Profile

# Every catalog offering is API-hosted in the MVP (self-hosted candidates are v1), so the
# only hosting option an offering can satisfy is this one.
API_HOSTING = "api"


@dataclass(frozen=True)
class Constraints:
    """What the partner requires, apart from what their traffic looks like."""

    # Providers the partner allows candidates to be called on. An EMPTY set drops every
    # offering: sending content to a provider that is not on the list is a counted data
    # incident, so "no list" must never mean "anything goes".
    allow_list: frozenset[str]
    hosting: str = API_HOSTING
    region: str | None = None  # None means no region requirement


# --- the checks: each returns the reason an offering fails, or None if it passes -------------


def check_incumbent(offering: Offering, profile: Profile) -> str | None:
    incumbent = profile.incumbent
    if incumbent and (offering.provider, offering.model) == (incumbent.provider, incumbent.model):
        return "it is the model the traffic already uses"
    return None


def check_context(offering: Offering, profile: Profile) -> str | None:
    need = profile.max_context_tokens
    if need is not None and offering.context_window < need:  # a window equal to the need passes
        return f"needs {need:,} tokens, window is {offering.context_window:,}"
    return None


def check_tools(offering: Offering, profile: Profile) -> str | None:
    if profile.uses_tools and not offering.tool_support:
        return "the traffic uses tools and it does not support them"
    return None


def check_region(offering: Offering, constraints: Constraints) -> str | None:
    region = constraints.region
    if region is not None and region not in offering.regions:
        return f"it does not serve region {region!r} (serves {', '.join(offering.regions)})"
    return None


def check_hosting(offering: Offering, constraints: Constraints) -> str | None:
    if constraints.hosting != API_HOSTING:
        return f"hosting option {constraints.hosting!r} is not available, every offering is {API_HOSTING}-hosted"
    return None


def check_allow_list(offering: Offering, constraints: Constraints) -> str | None:
    if offering.provider not in constraints.allow_list:
        return f"provider {offering.provider!r} is not on the allow-list"
    return None


# --- the filter ----------------------------------------------------------------------------


def stage_one(
    catalog: Catalog, profile: Profile, constraints: Constraints
) -> tuple[list[Offering], list[tuple[Offering, list[str]]]]:
    """Split the catalog into `(kept, dropped)`, both in catalog order.

    `dropped` pairs each dropped offering with every reason it failed, in a fixed order (the
    order of the checks in the module docstring). Every offering ends up in exactly one list.
    """
    kept: list[Offering] = []
    dropped: list[tuple[Offering, list[str]]] = []

    for offering in catalog.offerings:
        reasons = [
            reason
            for reason in (
                check_incumbent(offering, profile),
                check_context(offering, profile),
                check_tools(offering, profile),
                check_region(offering, constraints),
                check_hosting(offering, constraints),
                check_allow_list(offering, constraints),
            )
            if reason
        ]
        if reasons:
            dropped.append((offering, reasons))
        else:
            kept.append(offering)

    return kept, dropped
