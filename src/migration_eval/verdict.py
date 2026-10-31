"""Verdicts (EVL-8): one of four labels per candidate, from documented thresholds.

    safe to swap, swap with caveats, do not swap, insufficient data

The inputs are the candidate's agreement with its interval (EVL-6), the noise floor (EVL-1),
whether it was stopped early (EVL-2), how many requests ran, and whether it used converted
requests (EVL-10). The thresholds are plain numbers in `Thresholds`, written down and shown in
the report, so a verdict can be explained and re-run.

Open decisions (see the notes on EVL-8):
    * the numbers themselves, and whether they differ by component type (a classifier
      against a free-text generator). The ERD plans to tune them on the first two partners, so
      the defaults below are placeholders.
    * how to read the interval against the noise floor: "lower bound at or above the floor"
      is the strict reading used as the starting point
    * a candidate that used converted requests is held to "swap with caveats" at best (starting
      point), because its score also reflects the conversion
    * a "weak" category is one whose interval is entirely below the noise floor; categories
      smaller than `min_category_requests` are reported but never decide a verdict
    * a candidate with many failed replays should probably be "insufficient data"; the failure
      count is not an input yet
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from migration_eval.agreement import Agreement
from migration_eval.early_sample import CandidateRun


class Verdict(str, Enum):
    SAFE_TO_SWAP = "safe to swap"
    SWAP_WITH_CAVEATS = "swap with caveats"
    DO_NOT_SWAP = "do not swap"
    INSUFFICIENT_DATA = "insufficient data"


@dataclass(frozen=True)
class Thresholds:
    min_requests: int = 50  # fewer than this and there is no verdict
    safe_margin: float = 0.0  # lower bound must be >= noise_floor - safe_margin for "safe"
    caveat_margin: float = 0.05  # upper bound must be >= noise_floor - caveat_margin for "caveats"
    min_category_requests: int = 10  # smaller categories are reported but never decide a verdict


def _weak_categories(by_category: dict[str, Agreement], noise_floor: float, thresholds: Thresholds) -> list[str]:
    # Large enough to trust, and even its best plausible agreement is below the floor.
    return sorted(
        name
        for name, agreement in by_category.items()
        if agreement.n >= thresholds.min_category_requests
        and agreement.ci_high is not None
        and agreement.ci_high < noise_floor - thresholds.safe_margin
    )


def _caveats(
    overall: Agreement,
    by_category: dict[str, Agreement],
    noise_floor: float,
    thresholds: Thresholds,
    used_converted_requests: bool,
) -> list[str]:
    """Why a candidate that is not clearly bad is not clearly safe either. Empty means safe."""
    reasons = []
    if overall.ci_low < noise_floor - thresholds.safe_margin:
        reasons.append(f"the overall lower bound {overall.ci_low:.1%} is below the noise floor {noise_floor:.1%}")
    weak = _weak_categories(by_category, noise_floor, thresholds)
    if weak:
        reasons.append("it is below the noise floor for: " + ", ".join(weak))
    if used_converted_requests:
        reasons.append("it used converted requests, so the score reflects the conversion as well as the model")
    return reasons


def assign_verdict(
    run: CandidateRun,
    overall: Agreement,
    by_category: dict[str, Agreement],
    noise_floor: float,
    thresholds: Thresholds = Thresholds(),
    used_converted_requests: bool = False,
) -> Verdict:
    """The candidate's verdict.

    Starting rules: a stopped-early run is "do not swap"; fewer than `min_requests` run is
    "insufficient data"; an overall lower bound at the noise floor and every large-enough
    category too is "safe to swap"; a point estimate near the floor, or one weak category, is
    "swap with caveats"; otherwise "do not swap".
    """

    if run.stopped_early:
        return Verdict.DO_NOT_SWAP

    if run.requests_run < thresholds.min_requests or overall.ci_low is None:
        return Verdict.INSUFFICIENT_DATA

    if overall.ci_high < noise_floor - thresholds.caveat_margin:  # even its best case is clearly below
        return Verdict.DO_NOT_SWAP

    if _caveats(overall, by_category, noise_floor, thresholds, used_converted_requests):
        return Verdict.SWAP_WITH_CAVEATS

    return Verdict.SAFE_TO_SWAP


def explain_verdict(
    verdict: Verdict,
    run: CandidateRun,
    overall: Agreement,
    noise_floor: float,
    thresholds: Thresholds = Thresholds(),
    by_category: dict[str, Agreement] | None = None,
    used_converted_requests: bool = False,
) -> str:
    """One or two sentences naming the numbers behind the verdict, for the report."""

    if verdict is Verdict.INSUFFICIENT_DATA:
        return f"Only {run.requests_run} requests ran and at least {thresholds.min_requests} are needed."

    if verdict is Verdict.DO_NOT_SWAP and run.stopped_early:
        return (
            f"Stopped after {run.requests_run} requests: even its best plausible agreement was below "
            f"the incumbent's own agreement with itself ({noise_floor:.1%})."
        )

    numbers = f"Agreement {overall.rate:.1%} (95% CI {overall.ci_low:.1%} to {overall.ci_high:.1%}) over {overall.n} requests"

    if verdict is Verdict.DO_NOT_SWAP:
        return f"{numbers}; even the top of that range is clearly below the noise floor of {noise_floor:.1%}."

    if verdict is Verdict.SWAP_WITH_CAVEATS:
        reasons = _caveats(overall, by_category or {}, noise_floor, thresholds, used_converted_requests)
        return f"{numbers}, against a noise floor of {noise_floor:.1%}. Caveats: " + "; ".join(reasons) + "."

    return f"{numbers}; the lower bound is at or above the noise floor of {noise_floor:.1%}."
