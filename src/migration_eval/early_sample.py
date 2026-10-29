"""Early sample (EVL-2): run a candidate on 50 to 200 requests, stopping early if it is
clearly worse than the noise floor.

The candidate runs in batches on the shared sample. After each batch the agreement so far
gets a confidence interval; once enough requests have run, the candidate stops when the
interval's UPPER bound is below the noise floor, because even its best plausible agreement
is then lower than the incumbent's own. A stopped candidate later becomes "do not swap"
(EVL-8).

Agreement here is the candidate's output against the incumbent's replays (from EVL-1), using
the same comparator, so the candidate's score and the noise floor measure the same thing.

Open decision (see the notes on EVL-2): looking after every batch raises the chance of a
false stop, so interim looks use a stricter confidence level (STOP_CONFIDENCE) than the final
report's 95% (EVL-6).

A candidate that agrees with the incumbent on every request so far is never stopped, however
high the noise floor is: the interval's upper bound is exactly 1 in that case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

from migration_eval.models import CallRecord
from migration_eval.noise_floor import Comparator, same_output
from migration_eval.replay import CompleteFn, ReplayResult, replay

MIN_REQUESTS = 50  # never stop before this many requests have run
MAX_REQUESTS = 200  # never run more than this
BATCH_SIZE = 25
STOP_CONFIDENCE = 0.99


@dataclass(frozen=True)
class CandidateRun:
    provider: str
    model: str
    results: list[ReplayResult]  # lined up with the first `requests_run` records of the sample
    agreement: float | None  # share of requests judged equivalent to the incumbent; None if none ran
    requests_run: int
    stopped_early: bool  # True if it stopped before MAX_REQUESTS because it was clearly worse


def proportion_interval(agreed: int, n: int, confidence: float = STOP_CONFIDENCE) -> tuple[float, float]:
    """A confidence interval for a proportion: `agreed` out of `n`, as (lower, upper).

    A Wilson interval is a good fit for small samples and proportions near 0 or 1; the normal
    quantile comes from the standard library. Raise ValueError if `n` is not positive or
    `agreed` is outside 0..n.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= agreed <= n:
        raise ValueError("agreed must be between 0 and n")

    z = NormalDist().inv_cdf(0.5 + confidence / 2)  # two-sided: 0.99 leaves 0.5% in each tail
    p = agreed / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator

    # At the ends the bound is exactly 0 or 1; rounding must not leave 0.9999999 there, or a
    # candidate that agreed every time could look worse than a noise floor of 1.0.
    lower = 0.0 if agreed == 0 else max(0.0, centre - half_width)
    upper = 1.0 if agreed == n else min(1.0, centre + half_width)
    return lower, upper


def should_stop(
    agreed: int,
    n: int,
    noise_floor: float,
    min_requests: int = MIN_REQUESTS,
    confidence: float = STOP_CONFIDENCE,
) -> bool:
    """True when `n >= min_requests` and the interval's upper bound is below `noise_floor`."""
    if n < min_requests:
        return False
    _, upper = proportion_interval(agreed, n, confidence)
    return upper < noise_floor


def count_agreements(
    candidate_results: list[ReplayResult],
    incumbent_runs: list[list[ReplayResult]],
    comparator: Comparator = same_output,
) -> int:
    """How many of the candidate's results agree with the incumbent.

    `incumbent_runs` has one list per run, each lined up with `candidate_results`. Compare each
    candidate result with the incumbent's result for the same request in every run, and count
    the request as agreeing when most of those comparisons agree. (Open: the rule for
    combining the runs; majority is a starting point.)
    """
    agreed = 0
    for i, candidate in enumerate(candidate_results):
        incumbent_results = [run[i] for run in incumbent_runs]
        votes = sum(comparator(candidate, incumbent) for incumbent in incumbent_results)
        if votes * 2 > len(incumbent_results):
            agreed += 1
    return agreed


async def run_candidate(
    provider: str,
    model: str,
    sample: list[CallRecord],
    incumbent_runs: list[list[ReplayResult]],
    noise_floor: float,
    complete: CompleteFn,
    comparator: Comparator = same_output,
    *,
    batch_size: int = BATCH_SIZE,
    min_requests: int = MIN_REQUESTS,
    max_requests: int = MAX_REQUESTS,
    **replay_options: object,
) -> CandidateRun:
    """Replay the candidate on the sample in batches, stopping early when `should_stop`.

    Use the first `max_requests` records of `sample` (the same ones every candidate sees).
    After each batch, count agreements so far and check `should_stop`. Pass `replay_options`
    (speed, limiter, ...) through to `replay.replay`.
    """
    requests = sample[:max_requests]
    results: list[ReplayResult] = []
    agreed = 0
    stopped_early = False

    for start in range(0, len(requests), batch_size):
        end = min(start + batch_size, len(requests))
        batch_results = await replay(requests[start:end], provider, model, complete, **replay_options)
        results.extend(batch_results)
        agreed += count_agreements(batch_results, [run[start:end] for run in incumbent_runs], comparator)

        if end < len(requests) and should_stop(agreed, len(results), noise_floor, min_requests):
            stopped_early = True
            break

    return CandidateRun(
        provider=provider,
        model=model,
        results=results,
        agreement=agreed / len(results) if results else None,
        requests_run=len(results),
        stopped_early=stopped_early,
    )
