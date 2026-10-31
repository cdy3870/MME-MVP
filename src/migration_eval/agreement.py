"""Agreement report (EVL-6): agreement overall and per request category, with 95% bootstrap
confidence intervals.

Input is one yes/no per request ("the candidate agrees with the incumbent": a failed
deterministic check is a no, a judged pair uses `judgment_agrees`). Each figure comes with an
interval from resampling those outcomes, so a reader can tell 90% of 20 requests from 90% of
200.

The interim stopping rule in `early_sample.py` uses a Wilson interval at 99%; this final
report uses a bootstrap at 95%, as the requirement says.

Open decisions (see the notes on EVL-6):
    * where a request's category comes from. The ERD says the profile derives it from the
      upload; the schema offers `metadata` (caller tags), and `component` is the fallback.
    * numpy/scipy are named in the ticket; plain `random.Random(seed)` is enough for a few
      hundred outcomes and adds no dependency, so that is the starting point
    * a category with very few requests has an interval too wide to mean anything; the report
      should show the count and may need a minimum size

Every function here is a stub: implement them one at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from migration_eval.models import CallRecord

RESAMPLES = 10_000
REPORT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class Agreement:
    agreed: int
    n: int
    rate: float | None  # None when n == 0
    ci_low: float | None
    ci_high: float | None


def bootstrap_interval(
    outcomes: list[bool],
    confidence: float = REPORT_CONFIDENCE,
    resamples: int = RESAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """A percentile bootstrap interval for the share of True outcomes: resample with
    replacement `resamples` times, take the rate of each, and return the middle `confidence`.
    A fixed `seed` makes a report reproducible. Raise ValueError for no outcomes.
    """

    if not outcomes:
        raise ValueError("no outcomes")
    if len(outcomes) == 1:
        return 0.0, 1.0  # one observation says nothing; scipy needs at least two

    # "percentile", not BCa: BCa divides by zero when every outcome is the same, which is the
    # normal case for a candidate that agreed on every request.
    result = stats.bootstrap(
        data=(np.array(outcomes, dtype=float),),
        statistic=np.mean,
        n_resamples=resamples,
        confidence_level=confidence,
        method="percentile",
        rng=np.random.default_rng(seed),
    )

    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def agreement(outcomes: list[bool], seed: int = 0) -> Agreement:
    """The rate and its interval; all fields None except `agreed` and `n` when there are none."""
    if not outcomes:
        return Agreement(agreed=0, n=0, rate=None, ci_low=None, ci_high=None)

    count = sum(1 for o in outcomes if o)
    low, high = bootstrap_interval(outcomes, seed=seed)
    return Agreement(agreed=count, n=len(outcomes), rate=count / len(outcomes), ci_low=low, ci_high=high)


def request_category(record: CallRecord) -> str:
    """The category a request belongs to: `metadata["category"]` if the caller tagged one,
    otherwise the record's component.
    """

    category = (record.metadata or {}).get("category")
    if category is not None:
        return str(category)

    return record.component


def agreement_by_category(records: list[CallRecord], outcomes: list[bool], seed: int = 0) -> dict[str, Agreement]:
    """`agreement` for each category, from `outcomes` lined up with `records`. Raise
    ValueError if the lengths differ.
    """

    if len(records) != len(outcomes):
        raise ValueError("Lengths of records and outcomes differ")

    by_category: dict[str, list[bool]] = {}
    for record, outcome in zip(records, outcomes):
        by_category.setdefault(request_category(record), []).append(outcome)

    return {category: agreement(category_outcomes, seed) for category, category_outcomes in by_category.items()}



