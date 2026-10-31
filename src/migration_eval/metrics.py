"""Latency and cost per candidate (EVL-7), measured during the replay.

Latency is the time each replayed call took (`ReplayResult.latency_ms`, set by the real
`complete` function around the send); cost is the replayed token counts at the candidate's
catalog prices (`pricing.request_cost`). The same two numbers for the incumbent's own replays
give the baseline: compare the candidate with the incumbent *replayed now*, not with the
logged figures, because load, region and time of day change latency.

Open decisions (see the notes on EVL-7):
    * latency is only comparable at `speed=1` (a faster replay puts more requests in flight
      at once than production had), so a latency run should say what speed it used
    * a failed call has no meaningful latency or cost; count it separately, do not average it in
    * cached tokens and tokenizer differences, as in `pricing.py`
    * the spend meter (RUN-5) will want the running total, not only the final one

Every function here is a stub: implement them one at a time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from migration_eval.catalog import Offering
from migration_eval.pricing import request_cost
from migration_eval.replay import ReplayResult


@dataclass(frozen=True)
class LatencySummary:
    p50_ms: float | None  # None when no call succeeded
    p95_ms: float | None
    n: int  # calls the percentiles are over
    failed: int  # calls with an error, left out of the percentiles


@dataclass(frozen=True)
class CandidateMetrics:
    latency: LatencySummary
    cost_usd: float  # actual cost of the replayed calls
    cost_per_request_usd: float | None  # None when no call succeeded


def percentile(values: list[float], p: float) -> float:
    """The `p`th percentile (0-100) by linear interpolation. Raise ValueError for no values or
    a `p` outside 0-100.
    """

    if not values:
        raise ValueError("no values")
    if p < 0 or p > 100:
        raise ValueError("P out of bounds")

    sorted_vals = sorted(values)
    position = (len(sorted_vals) - 1) * p / 100  # 0 is the smallest value, len - 1 the largest
    below = math.floor(position)
    above = math.ceil(position)

    return sorted_vals[below] + (sorted_vals[above] - sorted_vals[below]) * (position - below)


def latency_summary(results: list[ReplayResult]) -> LatencySummary:
    """p50 and p95 of `latency_ms` over the successful results that have one."""

    latencies = [res.latency_ms for res in results if res.error is None and res.latency_ms is not None]
    failed = sum(1 for res in results if res.error is not None)

    if not latencies:
        return LatencySummary(p50_ms=None, p95_ms=None, n=0, failed=failed)
    return LatencySummary(p50_ms=percentile(latencies, 50), p95_ms=percentile(latencies, 95), n=len(latencies), failed=failed)


def _priced(results: list[ReplayResult]) -> list[ReplayResult]:
    # Failed calls and calls without both token counts have no cost to add.
    return [r for r in results if r.error is None and r.input_tokens is not None and r.output_tokens is not None]


def replay_cost(offering: Offering, results: list[ReplayResult]) -> float:
    """USD for the replayed calls: `request_cost` over each result with both token counts."""

    return sum(request_cost(offering, r.input_tokens, r.output_tokens) for r in _priced(results))


def candidate_metrics(offering: Offering, results: list[ReplayResult]) -> CandidateMetrics:
    """Latency summary and cost together."""

    cost = replay_cost(offering, results)
    priced = len(_priced(results))

    return CandidateMetrics(
        latency=latency_summary(results),
        cost_usd=cost,
        cost_per_request_usd=cost / priced if priced else None,
    )
