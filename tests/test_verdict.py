"""Verdicts (EVL-8): four labels from documented thresholds."""

from __future__ import annotations

import pytest

from migration_eval.agreement import Agreement
from migration_eval.early_sample import CandidateRun
from migration_eval.verdict import Thresholds, Verdict, assign_verdict, explain_verdict

FLOOR = 0.90


def run(requests_run=100, stopped_early=False) -> CandidateRun:
    return CandidateRun("openai", "cand", [], 0.9, requests_run, stopped_early)


def agree(rate, low, high, n=100) -> Agreement:
    return Agreement(agreed=round(rate * n), n=n, rate=rate, ci_low=low, ci_high=high)


GOOD = agree(0.95, 0.91, 0.98)  # lower bound above the floor
NEAR = agree(0.88, 0.80, 0.94)  # interval straddles the floor
BAD = agree(0.60, 0.50, 0.70)  # best case far below the floor


def verdict(overall, by_category=None, **kwargs):
    run_kwargs = {k: kwargs.pop(k) for k in ("requests_run", "stopped_early") if k in kwargs}
    return assign_verdict(run(**run_kwargs), overall, by_category or {}, FLOOR, **kwargs)


# --- the four labels ----------------------------------------------------------------------


def test_clearly_at_or_above_the_floor_is_safe():
    assert verdict(GOOD) is Verdict.SAFE_TO_SWAP


def test_an_interval_straddling_the_floor_is_swap_with_caveats():
    assert verdict(NEAR) is Verdict.SWAP_WITH_CAVEATS


def test_clearly_below_the_floor_is_do_not_swap():
    assert verdict(BAD) is Verdict.DO_NOT_SWAP


def test_too_few_requests_is_insufficient_data():
    assert verdict(GOOD, requests_run=49) is Verdict.INSUFFICIENT_DATA
    assert verdict(GOOD, requests_run=50) is Verdict.SAFE_TO_SWAP


def test_a_stopped_early_run_is_do_not_swap_even_with_a_good_looking_score():
    assert verdict(GOOD, stopped_early=True) is Verdict.DO_NOT_SWAP


def test_a_run_with_no_interval_is_insufficient_data_not_a_crash():
    nothing = Agreement(agreed=0, n=0, rate=None, ci_low=None, ci_high=None)
    assert verdict(nothing, requests_run=100) is Verdict.INSUFFICIENT_DATA


def test_the_function_never_returns_none():
    for overall in (GOOD, NEAR, BAD, agree(0.89, 0.85, 0.89)):
        assert isinstance(verdict(overall), Verdict)


# --- the thresholds -----------------------------------------------------------------------


def test_boundaries_are_inclusive():
    assert verdict(agree(0.95, FLOOR, 0.98)) is Verdict.SAFE_TO_SWAP  # lower bound exactly at the floor
    assert verdict(agree(0.80, 0.70, FLOOR - 0.05)) is Verdict.SWAP_WITH_CAVEATS  # upper bound at floor - margin


def test_a_safe_margin_allows_a_lower_bound_a_little_under_the_floor():
    almost = agree(0.93, 0.88, 0.97)
    assert verdict(almost) is Verdict.SWAP_WITH_CAVEATS
    assert verdict(almost, thresholds=Thresholds(safe_margin=0.05)) is Verdict.SAFE_TO_SWAP


def test_the_minimum_request_count_can_be_changed():
    assert verdict(GOOD, requests_run=20, thresholds=Thresholds(min_requests=20)) is Verdict.SAFE_TO_SWAP


# --- categories ---------------------------------------------------------------------------


def test_a_weak_large_category_downgrades_safe_to_caveats():
    categories = {"chat": agree(0.97, 0.93, 0.99), "tools": agree(0.60, 0.47, 0.72, n=60)}
    assert verdict(GOOD, categories) is Verdict.SWAP_WITH_CAVEATS


def test_a_small_weak_category_never_decides_a_verdict():
    categories = {"chat": agree(0.97, 0.93, 0.99), "rare": agree(0.40, 0.15, 0.70, n=5)}
    assert verdict(GOOD, categories) is Verdict.SAFE_TO_SWAP


def test_a_category_whose_interval_still_reaches_the_floor_is_not_weak():
    categories = {"tools": agree(0.80, 0.60, 0.92, n=30)}
    assert verdict(GOOD, categories) is Verdict.SAFE_TO_SWAP


# --- converted requests -------------------------------------------------------------------


def test_converted_requests_cap_the_verdict_at_caveats():
    assert verdict(GOOD, used_converted_requests=True) is Verdict.SWAP_WITH_CAVEATS


def test_converted_requests_do_not_rescue_or_worsen_other_verdicts():
    assert verdict(BAD, used_converted_requests=True) is Verdict.DO_NOT_SWAP
    assert verdict(GOOD, requests_run=10, used_converted_requests=True) is Verdict.INSUFFICIENT_DATA


# --- explain_verdict ----------------------------------------------------------------------


def test_every_verdict_has_an_explanation_with_its_numbers():
    safe = explain_verdict(Verdict.SAFE_TO_SWAP, run(), GOOD, FLOOR)
    assert "95.0%" in safe and "90.0%" in safe and "100 requests" in safe

    bad = explain_verdict(Verdict.DO_NOT_SWAP, run(), BAD, FLOOR)
    assert "60.0%" in bad and "below" in bad

    few = explain_verdict(Verdict.INSUFFICIENT_DATA, run(requests_run=12), GOOD, FLOOR)
    assert "12" in few and "50" in few

    stopped = explain_verdict(Verdict.DO_NOT_SWAP, run(requests_run=50, stopped_early=True), BAD, FLOOR)
    assert "Stopped after 50" in stopped


def test_caveats_are_named_in_the_explanation():
    categories = {"tools": agree(0.60, 0.47, 0.72, n=60)}
    text = explain_verdict(Verdict.SWAP_WITH_CAVEATS, run(), GOOD, FLOOR, by_category=categories, used_converted_requests=True)
    assert "tools" in text and "conversion" in text


def test_the_straddling_case_explains_the_lower_bound():
    text = explain_verdict(Verdict.SWAP_WITH_CAVEATS, run(), NEAR, FLOOR)
    assert "80.0%" in text and "below the noise floor" in text


@pytest.mark.parametrize("overall", [GOOD, NEAR, BAD])
def test_assign_and_explain_agree_on_every_case(overall):
    v = verdict(overall)
    assert isinstance(explain_verdict(v, run(), overall, FLOOR), str)
