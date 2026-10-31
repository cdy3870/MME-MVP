"""Agreement report (EVL-6): rates and bootstrap intervals overall and per category."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from migration_eval.agreement import agreement, agreement_by_category, bootstrap_interval, request_category
from migration_eval.models import CallRecord


def rec(i=0, component="c", metadata=None) -> CallRecord:
    body = {
        "schema_version": "v0", "id": f"r{i}", "component": component, "provider": "openai", "model": "m",
        "started_at": datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc).isoformat(),
    }  # fmt: skip
    if metadata is not None:
        body["metadata"] = metadata
    return CallRecord.model_validate(body)


# --- bootstrap_interval -------------------------------------------------------------------


def test_the_interval_contains_the_observed_rate():
    outcomes = [True] * 40 + [False] * 10
    low, high = bootstrap_interval(outcomes)
    assert 0 <= low <= 0.8 <= high <= 1


def test_the_interval_is_reproducible_with_a_seed():
    outcomes = [True] * 40 + [False] * 10
    assert bootstrap_interval(outcomes, seed=1) == bootstrap_interval(outcomes, seed=1)


def test_more_requests_give_a_narrower_interval():
    small = bootstrap_interval([True] * 8 + [False] * 2)
    large = bootstrap_interval([True] * 80 + [False] * 20)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_every_outcome_the_same_gives_a_point_interval_not_nan():
    assert bootstrap_interval([True] * 50) == (1.0, 1.0)
    assert bootstrap_interval([False] * 50) == (0.0, 0.0)


def test_one_outcome_says_nothing():
    assert bootstrap_interval([True]) == (0.0, 1.0)


def test_no_outcomes_is_an_error():
    with pytest.raises(ValueError):
        bootstrap_interval([])


# --- agreement ----------------------------------------------------------------------------


def test_agreement_counts_and_rate():
    result = agreement([True, True, True, False])
    assert (result.agreed, result.n, result.rate) == (3, 4, 0.75)
    assert result.ci_low <= 0.75 <= result.ci_high


def test_agreement_with_no_outcomes_is_empty_not_a_crash():
    result = agreement([])
    assert (result.agreed, result.n, result.rate, result.ci_low, result.ci_high) == (0, 0, None, None, None)


def test_agreement_with_a_single_outcome():
    result = agreement([True])
    assert result.rate == 1.0 and (result.ci_low, result.ci_high) == (0.0, 1.0)


# --- request_category ---------------------------------------------------------------------


def test_a_tagged_category_wins():
    assert request_category(rec(component="bot", metadata={"category": "refund"})) == "refund"


def test_other_metadata_keys_are_not_a_category():
    assert request_category(rec(component="bot", metadata={"region": "us"})) == "bot"


def test_no_metadata_falls_back_to_the_component():
    assert request_category(rec(component="bot")) == "bot"


def test_a_non_string_category_is_turned_into_text():
    record = rec(metadata={"category": 3})
    assert request_category(record) == str(record.metadata["category"])


# --- agreement_by_category ----------------------------------------------------------------


def test_each_category_gets_only_its_own_outcomes():
    records = [rec(0, metadata={"category": "chat"}), rec(1, metadata={"category": "chat"}), rec(2, metadata={"category": "tools"})]
    result = agreement_by_category(records, [True, True, False])
    assert set(result) == {"chat", "tools"}
    assert (result["chat"].agreed, result["chat"].n) == (2, 2)
    assert (result["tools"].agreed, result["tools"].n) == (0, 1)


def test_categories_default_to_the_component():
    records = [rec(0, component="a"), rec(1, component="b"), rec(2, component="a")]
    result = agreement_by_category(records, [True, False, False])
    assert (result["a"].agreed, result["a"].n) == (1, 2) and (result["b"].agreed, result["b"].n) == (0, 1)


def test_no_records_gives_no_categories():
    assert agreement_by_category([], []) == {}


def test_records_and_outcomes_must_line_up():
    with pytest.raises(ValueError, match="differ"):
        agreement_by_category([rec(0)], [True, False])
