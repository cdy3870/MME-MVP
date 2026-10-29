"""Noise floor (EVL-1): how much the incumbent disagrees with itself.

A candidate's differences from the incumbent only mean something compared with the
incumbent's own run-to-run variation. This replays the incumbent several times on the sample
and measures how often those replays agree with each other.

"Agree" is decided by a comparator passed in. The real one comes from EVL-4 (deterministic
checks) and EVL-5 (the judge); `same_output` below is a simple stand-in so this can be built
and tested first.

Open decisions (see the notes on EVL-1):
    * the exact definition of "agreement" (equivalent only, or equivalent plus better) is
      shared with EVL-5 and EVL-6 and is not settled
    * a replay that failed (an `error`) counts as disagreeing, because `same_output` says it
      agrees with nothing. That lowers the noise floor, which makes candidates look better
      than they are, so a run with many failures should not be trusted.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

from migration_eval.models import CallRecord
from migration_eval.replay import CompleteFn, ReplayResult, replay

# True when two outputs for the same request count as equivalent.
Comparator = Callable[[ReplayResult, ReplayResult], bool]

NOISE_FLOOR_RUNS = 3


def _calls(result: ReplayResult) -> list[tuple]:
    # Names and arguments only: tool-call ids differ on every call, so comparing them
    # would make identical answers disagree.
    return [(c.get("name"), c.get("arguments"), c.get("arguments_raw")) for c in result.tool_calls or []]


def same_output(a: ReplayResult, b: ReplayResult) -> bool:
    """A stand-in comparator: the text is equal after trimming whitespace and the tool calls
    (names and arguments) are equal. A result with an `error` agrees with nothing.
    """
    if a.error is not None or b.error is not None:
        return False

    return (a.text or "").strip() == (b.text or "").strip() and _calls(a) == _calls(b)


async def replay_incumbent(
    sample: list[CallRecord],
    provider: str,
    model: str,
    complete: CompleteFn,
    runs: int = NOISE_FLOOR_RUNS,
    **replay_options: object,
) -> list[list[ReplayResult]]:
    """Replay the incumbent on the sample `runs` times with the logged settings.

    Returns one list of results per run, each lined up with `sample`. Run them one after
    another (not interleaved) and pass `replay_options` (speed, limiter, ...) through to
    `replay.replay`. Keep the outputs: candidates are compared against them, and EVL-9 will
    cache them.
    """
    results = []

    for _ in range(runs):
        results.append(await replay(sample, provider, model, complete, **replay_options))

    return results


def noise_floor(runs: list[list[ReplayResult]], comparator: Comparator = same_output) -> float | None:
    """The incumbent's agreement with itself: the share of comparisons judged equivalent.

    For each request, compare every pair of runs (3 runs give 1-2, 1-3 and 2-3), and return
    agreeing comparisons divided by all comparisons. Returns None when there is nothing to
    compare (fewer than two runs, or no requests). Runs must be the same length.
    """
    if len(runs) < 2 or not runs[0]:
        return None
    if len({len(run) for run in runs}) != 1:
        raise ValueError("every run must have the same number of results")

    agreeing_comparisons = 0
    total_comparisons = 0
    for results in zip(*runs):  # the results for one request, one from each run
        for a, b in itertools.combinations(results, 2):
            total_comparisons += 1
            if comparator(a, b):
                agreeing_comparisons += 1

    return agreeing_comparisons / total_comparisons
