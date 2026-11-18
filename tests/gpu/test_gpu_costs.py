"""GPU-7: cost estimates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from migration_eval.gpu.assessment import assess
from migration_eval.gpu.costs import HOURS_PER_MONTH, consolidate, estimate_costs, monthly_cost, savings, tokens_per_hour
from migration_eval.gpu.labels import Estimate
from migration_eval.gpu.prices import PriceTable
from migration_eval.models import CallRecord

# tests/ has no __init__.py files, so pytest puts this directory on sys.path (prepend mode).
from test_gpu_assessment import profile, sample
from test_gpu_prices import H100


T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)


def record(i: int, at: datetime, tokens: int | None = 1000) -> CallRecord:
    usage = {"input_tokens": tokens // 2, "output_tokens": tokens // 2} if tokens else {}
    return CallRecord.model_validate(
        {
            "schema_version": "v0", "id": f"r{i}", "component": "c", "provider": "openai",
            "model": "m", "started_at": at.isoformat(), **({"usage": usage} if usage else {}),
        }  # fmt: skip
    )


def test_monthly_cost_is_a_range():
    cost = monthly_cost(2, H100)
    assert (cost.low, cost.high, cost.basis) == (2 * 2.0 * HOURS_PER_MONTH, 2 * 4.0 * HOURS_PER_MONTH, "estimated")


def test_two_lightly_used_gpus_consolidate_to_one():
    p = profile([sample(0, i, 30) for i in range(10)] + [sample(1, i, 30) for i in range(10)])
    c = consolidate(assess(p))
    assert c.current_gpus == 2 and c.needed_gpus.high == 1


def test_tokens_per_hour_skips_records_without_counts():
    records = [record(0, T0), record(1, T0 + timedelta(minutes=30), None), record(2, T0 + timedelta(hours=1))]
    assert tokens_per_hour(records) == pytest.approx(2000)


def test_savings_spans_worst_to_best_case():
    monthly, annual = savings(Estimate(low=100, high=200, unit="USD/month"), Estimate(low=50, high=80, unit="USD/month"))
    assert (monthly.low, monthly.high) == (20, 150)
    assert annual.high == 150 * 12


def test_unpriced_gpu_is_reported_missing():
    p = profile([sample(0, i, 30) for i in range(10)] + [sample(1, i, 30) for i in range(10)])
    costs = estimate_costs(assess(p), p, PriceTable(prices=[]))
    assert costs.current_monthly_cost is None and costs.missing


def test_estimate_costs_with_prices_and_records():
    p = profile([sample(g, i, 30) for g in (0, 1) for i in range(10)])
    records = [record(0, T0, 3_600_000), record(1, T0 + timedelta(hours=1), 3_600_000)]
    costs = estimate_costs(assess(p), p, PriceTable(prices=[H100.model_copy(update={"match": ["H100"]})]), records)
    assert costs.current_monthly_cost.low == 2 * 2.0 * HOURS_PER_MONTH
    assert costs.consolidation.needed_gpus.high == 1
    assert costs.sizing.gpu == "H100 80GB"
    assert costs.monthly_savings.high > 0
    # 2 GPUs at $2-4/h = $4-8/h, over 7.2M tokens in one hour
    assert costs.cost_per_million_tokens.low == pytest.approx(4 / 7.2)
    assert not costs.missing
