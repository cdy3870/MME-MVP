"""Latency and cost per candidate (EVL-7)."""

from __future__ import annotations

import pytest

from migration_eval.catalog import Catalog
from migration_eval.metrics import candidate_metrics, latency_summary, percentile, replay_cost
from migration_eval.replay import ReplayResult


def offering(input_price=2.50, output_price=10.00):
    return Catalog.model_validate(
        {
            "offerings": [
                {
                    "provider": "openai", "model": "gpt-4o", "context_window": 128000, "tool_support": True,
                    "regions": ["us"],
                    "prices_usd_per_million_tokens": {"input": input_price, "output": output_price},
                    "as_of": "2026-10-24", "source_url": "https://example.com/p",
                }
            ]
        }  # fmt: skip
    ).offerings[0]


def result(latency=100.0, input_tokens=1000, output_tokens=500, error=None) -> ReplayResult:
    return ReplayResult("ok", None, input_tokens, output_tokens, latency, error)


# --- percentile ---------------------------------------------------------------------------


def test_percentile_of_one_value_is_that_value():
    assert percentile([7.0], 50) == 7.0 and percentile([7.0], 95) == 7.0


def test_percentile_interpolates_between_values():
    assert percentile([10, 20, 30, 40], 50) == 25.0
    assert percentile([0, 100], 95) == pytest.approx(95.0)


def test_percentile_does_not_need_sorted_input():
    assert percentile([40, 10, 30, 20], 50) == 25.0


def test_percentile_ends():
    assert percentile([3, 1, 2], 0) == 1 and percentile([3, 1, 2], 100) == 3


@pytest.mark.parametrize("values, p", [([], 50), ([1, 2], -1), ([1, 2], 101)])
def test_percentile_rejects_bad_input(values, p):
    with pytest.raises(ValueError):
        percentile(values, p)


# --- latency_summary ----------------------------------------------------------------------


def test_p50_and_p95_over_successful_calls():
    results = [result(latency=float(ms)) for ms in range(1, 101)]  # 1..100 ms
    summary = latency_summary(results)
    assert summary.p50_ms == pytest.approx(50.5) and summary.p95_ms == pytest.approx(95.05)
    assert summary.n == 100 and summary.failed == 0


def test_failed_calls_are_counted_but_left_out_of_the_percentiles():
    summary = latency_summary([result(100.0), result(200.0), result(5000.0, error="HTTP 500")])
    assert summary.n == 2 and summary.failed == 1 and summary.p95_ms < 300


def test_calls_without_a_latency_are_skipped():
    summary = latency_summary([result(None), result(120.0)])
    assert summary.n == 1 and summary.p50_ms == 120.0


def test_no_successful_calls_gives_none_not_a_crash():
    assert latency_summary([]) .p50_ms is None
    summary = latency_summary([result(error="boom")])
    assert (summary.p50_ms, summary.p95_ms, summary.n, summary.failed) == (None, None, 0, 1)


# --- replay_cost --------------------------------------------------------------------------


def test_cost_is_tokens_at_the_offering_prices():
    # 1000 in at $2.50/M + 500 out at $10/M = 0.0025 + 0.005 = 0.0075 per call
    assert replay_cost(offering(), [result(), result()]) == pytest.approx(0.015)


def test_calls_that_failed_or_lack_token_counts_cost_nothing():
    results = [result(), result(error="boom"), result(input_tokens=None), result(output_tokens=None)]
    assert replay_cost(offering(), results) == pytest.approx(0.0075)


def test_no_results_costs_zero():
    assert replay_cost(offering(), []) == 0.0


# --- candidate_metrics --------------------------------------------------------------------


def test_metrics_put_latency_and_cost_together():
    metrics = candidate_metrics(offering(), [result(100.0), result(300.0)])
    assert metrics.latency.p50_ms == 200.0 and metrics.latency.n == 2
    assert metrics.cost_usd == pytest.approx(0.015) and metrics.cost_per_request_usd == pytest.approx(0.0075)


def test_cost_per_request_averages_over_priced_calls_only():
    metrics = candidate_metrics(offering(), [result(), result(error="boom")])
    assert metrics.cost_per_request_usd == pytest.approx(0.0075)


def test_metrics_with_nothing_succeeded():
    metrics = candidate_metrics(offering(), [result(error="boom")])
    assert metrics.cost_usd == 0.0 and metrics.cost_per_request_usd is None and metrics.latency.p50_ms is None
