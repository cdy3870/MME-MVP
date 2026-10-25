"""Cost prediction (CAT-4), the dropped-candidate listing (CAT-5)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from migration_eval.candidates import dropped_for_report, format_dropped, stage_two
from migration_eval.catalog import Catalog
from migration_eval.models import CallRecord
from migration_eval.pricing import predicted_monthly_cost, request_cost


def offering(model="gpt-4o", provider="openai", input_price=2.50, output_price=10.00):
    return Catalog.model_validate(
        {
            "offerings": [
                {
                    "provider": provider,
                    "model": model,
                    "context_window": 128000,
                    "tool_support": True,
                    "regions": ["us"],
                    "prices_usd_per_million_tokens": {"input": input_price, "output": output_price},
                    "as_of": "2026-10-24",
                    "source_url": "https://example.com/p",
                }
            ]
        }
    ).offerings[0]


def record(input_tokens=1000, output_tokens=500) -> CallRecord:
    usage = {}
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    return CallRecord.model_validate(
        {
            "schema_version": "v0", "id": "r", "component": "c", "provider": "openai",
            "model": "m", "started_at": "2026-10-24T00:00:00Z", **({"usage": usage} if usage else {}),
        }  # fmt: skip
    )


DAY, WEEK = timedelta(days=1), timedelta(days=7)

# --- request_cost -------------------------------------------------------------------------


def test_input_and_output_tokens_are_priced_separately():
    assert request_cost(offering(), 1_000_000, 1_000_000) == pytest.approx(12.50)  # 2.50 + 10.00
    assert request_cost(offering(), 1000, 500) == pytest.approx(0.0075)  # 0.0025 + 0.005
    assert request_cost(offering(), 0, 0) == 0


def test_a_free_model_costs_nothing():
    assert request_cost(offering(input_price=0, output_price=0), 10**9, 10**9) == 0


# --- predicted_monthly_cost ---------------------------------------------------------------


def test_the_logged_cost_is_scaled_from_the_window_to_a_month():
    one_call = 0.0075
    assert predicted_monthly_cost(offering(), [record()], DAY) == pytest.approx(one_call * 30)
    assert predicted_monthly_cost(offering(), [record()], WEEK) == pytest.approx(one_call * 30 / 7)


def test_a_window_shorter_than_a_day_is_not_truncated():
    assert predicted_monthly_cost(offering(), [record()], timedelta(hours=12)) == pytest.approx(0.0075 * 30 / 0.5)
    assert predicted_monthly_cost(offering(), [record()], timedelta(hours=36)) == pytest.approx(0.0075 * 30 / 1.5)


def test_every_record_with_counts_is_added_up():
    assert predicted_monthly_cost(offering(), [record(), record(), record()], DAY) == pytest.approx(0.0075 * 3 * 30)


def test_records_without_both_token_counts_are_skipped():
    records = [record(), record(input_tokens=None, output_tokens=None), record(input_tokens=5, output_tokens=None)]
    assert predicted_monthly_cost(offering(), records, DAY) == pytest.approx(0.0075 * 30)


def test_nothing_to_predict_from_is_none():
    assert predicted_monthly_cost(offering(), [], DAY) is None
    assert predicted_monthly_cost(offering(), [record(input_tokens=None, output_tokens=None)], DAY) is None


@pytest.mark.parametrize("window", [timedelta(0), timedelta(days=-1)])
def test_a_window_that_is_not_positive_is_an_error(window):
    with pytest.raises(ValueError, match="positive"):
        predicted_monthly_cost(offering(), [record()], window)


# --- stage_two ----------------------------------------------------------------------------
# One call of 1000 + 500 tokens a day costs 0.0075 * 30 = $0.225 a month on gpt-4o.


def test_an_offering_over_the_cap_is_dropped_with_a_dollar_formatted_reason():
    big_call = record(input_tokens=10_000, output_tokens=5_000)  # $0.075 a call, $2.25 a month
    kept, dropped = stage_two([offering()], [big_call], DAY, cost_cap_usd=1.00)
    assert kept == []
    (_, reasons), = dropped
    assert reasons == ["predicted $2.25 a month, cap is $1.00"]


def test_an_offering_under_the_cap_is_kept():
    kept, dropped = stage_two([offering()], [record()], DAY, cost_cap_usd=1.00)
    assert len(kept) == 1 and dropped == []


def test_a_cost_exactly_equal_to_the_cap_passes():
    cost = predicted_monthly_cost(offering(), [record()], DAY)
    kept, dropped = stage_two([offering()], [record()], DAY, cost_cap_usd=cost)
    assert len(kept) == 1 and dropped == []


def test_the_cheaper_offerings_are_kept_and_the_dearer_dropped_in_input_order():
    cheap, pricey = offering("cheap", input_price=0.1, output_price=0.4), offering("pricey", input_price=50, output_price=200)
    kept, dropped = stage_two([pricey, cheap], [record()], DAY, cost_cap_usd=0.05)
    assert [o.model for o in kept] == ["cheap"] and [o.model for o, _ in dropped] == ["pricey"]


def test_with_no_token_counts_nothing_can_be_predicted_so_everything_is_kept():
    kept, dropped = stage_two([offering()], [record(input_tokens=None, output_tokens=None)], DAY, cost_cap_usd=0.0)
    assert len(kept) == 1 and dropped == []


# --- dropped_for_report / format_dropped (CAT-5) ------------------------------------------


def test_every_dropped_candidate_keeps_its_reasons_stage_one_first():
    s1 = [(offering("a", "gemini"), ["provider 'gemini' is not on the allow-list", "it does not serve region 'eu'"])]
    s2 = [(offering("b"), ["predicted $5.00 a month, cap is $1.00"])]
    assert dropped_for_report(s1, s2) == [
        {"provider": "gemini", "model": "a", "stage": 1,
         "reasons": ["provider 'gemini' is not on the allow-list", "it does not serve region 'eu'"]},
        {"provider": "openai", "model": "b", "stage": 2, "reasons": ["predicted $5.00 a month, cap is $1.00"]},
    ]  # fmt: skip


def test_nothing_dropped_gives_an_empty_listing():
    assert dropped_for_report([], []) == []


def test_the_listing_is_plain_data_the_json_report_can_hold():
    import json

    entries = dropped_for_report([(offering(), ["a reason"])], [])
    assert json.loads(json.dumps(entries)) == entries


def test_the_text_listing_shows_a_heading_and_each_reason_indented():
    text = format_dropped(
        [
            {"provider": "openai", "model": "gpt-4o", "stage": 1, "reasons": ["too small", "no tools"]},
            {"provider": "gemini", "model": "flash", "stage": 2, "reasons": ["predicted $9.00 a month, cap is $1.00"]},
        ]
    )
    assert text.splitlines() == [
        "openai/gpt-4o (dropped in stage 1)",
        "  too small",
        "  no tools",
        "gemini/flash (dropped in stage 2)",
        "  predicted $9.00 a month, cap is $1.00",
    ]


def test_the_text_listing_says_so_when_nothing_was_dropped():
    assert format_dropped([]) == "No models were dropped."
