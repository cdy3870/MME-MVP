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

Stage 2 (predicted cost, CAT-4, `stage_two`) runs on what Stage 1 keeps. CAT-5
(`dropped_for_report`, `format_dropped`) turns both stages' drops into the listing the report
shows. The CLI `migration-eval candidates` and the `POST /candidates` endpoint come later,
once uploads exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from migration_eval.catalog import Catalog, Offering
from migration_eval.models import CallRecord
from migration_eval.profile import Profile

from migration_eval.pricing import predicted_monthly_cost


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


# --- Stage 2 (CAT-4) -----------------------------------------------------------------------


def stage_two(
    offerings: list[Offering],
    records: list[CallRecord],
    window: timedelta,
    cost_cap_usd: float,
) -> tuple[list[Offering], list[tuple[Offering, list[str]]]]:
    """Drop offerings whose predicted monthly cost is above `cost_cap_usd`.

    Run it on what `stage_one` kept, with the component's records and the period they cover
    (`window`, passed to `pricing.predicted_monthly_cost`). Returns `(kept, dropped)` in the
    same shape as `stage_one`, both in input order, with a reason such as
    "predicted $1,250.00 a month, cap is $1,000.00".

    A cost exactly equal to the cap passes. When no cost can be predicted (no token counts)
    the offering is kept, as in `check_context`: say so in the module docstring's open
    decisions if you keep that choice.
    """

    kept: list[Offering] = []
    dropped: list[tuple[Offering, list[str]]] = []

    for offering in offerings:
        monthly_cost = predicted_monthly_cost(offering, records, window)

        if monthly_cost is not None and monthly_cost > cost_cap_usd:
            reason = f"predicted ${monthly_cost:,.2f} a month, cap is ${cost_cap_usd:,.2f}"
            dropped.append((offering, [reason]))

        else:
            kept.append(offering)


    return kept, dropped

# --- Reporting the dropped candidates (CAT-5) ----------------------------------------------


def dropped_for_report(
    stage_one_dropped: list[tuple[Offering, list[str]]],
    stage_two_dropped: list[tuple[Offering, list[str]]],
) -> list[dict[str, Any]]:
    """Every dropped candidate with its reasons, as plain data the JSON report can hold.

    One dict per candidate: `{"provider": ..., "model": ..., "stage": 1 or 2, "reasons": [...]}`.
    Stage 1 drops come first, then stage 2, each in the order given. A candidate dropped in
    stage 1 never reaches stage 2, so none appears twice.
    """
    dropped = []
    for candidate in stage_one_dropped:
        dropped.append({"provider": candidate[0].provider, "model": candidate[0].model, "stage": 1, "reasons": candidate[1]})

    for candidate in stage_two_dropped:
        dropped.append({"provider": candidate[0].provider, "model": candidate[0].model, "stage": 2, "reasons": candidate[1]})

    return dropped

def format_dropped(dropped: list[dict[str, Any]]) -> str:
    """Render `dropped_for_report`'s output as text for the terminal and the HTML report.

    One block per candidate, a heading like `openai/gpt-4o (dropped in stage 1)` followed by
    each reason on its own indented line. Return a short sentence when nothing was dropped.
    """
    report = ""
    for cand in dropped:
        report += f"{cand['provider']}/{cand['model']} (dropped in stage {cand['stage']})\n"
        for reason in cand["reasons"]:
            report += f"  {reason}\n"

    if len(dropped) == 0:
        return "No models were dropped."

    return report

