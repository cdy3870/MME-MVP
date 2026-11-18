"""GPU-7: cost estimates (Deliverable 3).

Four figures, every one an Estimate range (GPU-8), built from the GPU-5 assessment, the
recorder logs and gpu_prices.yaml (GPU-10):

    current cost        GPU count x hourly price (low..high), per month
    consolidation       how many GPUs the measured load needs at TARGET_UTILIZATION_PCT
    cost per 1M tokens  GPU cost per hour / tokens served per hour (from the recorder logs)
    savings             current cost minus the consolidated cost, monthly and annual

Assumptions to state on each figure: the capture is representative of normal load, prices
as of the price entry's `as_of`, HOURS_PER_MONTH, and the utilization target.

Consolidation is deliberately simple: total measured busy time (sum of per-GPU mean
utilization) over the target, rounded up, never below what memory needs (summed peak memory
use over one GPU's memory). It ignores model-parallel constraints; say so. Sizing picks the
cheapest priced GPU type whose memory fits the per-GPU peak.

A GPU without a price entry, or no token counts in the logs, isn't an error: leave that
figure None and add the reason to `missing`.
"""

from __future__ import annotations

import math
from datetime import timedelta

from pydantic import BaseModel

from migration_eval.gpu.assessment import GpuAssessment
from migration_eval.gpu.labels import Estimate
from migration_eval.gpu.prices import GpuPrice, PriceTable, match_gpu
from migration_eval.gpu.profile import GpuProfile
from migration_eval.models import CallRecord

HOURS_PER_MONTH = 730
TARGET_UTILIZATION_PCT = (60.0, 75.0)  # the range consolidation aims for: high end packs tighter
MIN_TOKEN_SPAN = timedelta(minutes=10)  # shorter spans give a meaningless hourly rate

REPRESENTATIVE = "the capture reflects normal production load"


class Consolidation(BaseModel):
    current_gpus: int  # measured
    needed_gpus: Estimate  # unit "GPUs"
    limited_by: str  # "utilization" or "memory"


class Sizing(BaseModel):
    gpu: str  # GpuPrice.gpu of the suggested type
    count: Estimate  # unit "GPUs"
    monthly_cost: Estimate  # unit "USD/month"


class CostEstimates(BaseModel):
    current_monthly_cost: Estimate | None = None  # "USD/month"
    consolidation: Consolidation | None = None
    sizing: Sizing | None = None
    cost_per_million_tokens: Estimate | None = None  # "USD per 1M tokens"
    monthly_savings: Estimate | None = None  # "USD/month"; low may be negative
    annual_savings: Estimate | None = None  # "USD/year"
    missing: list[str] = []  # why a figure is None: "no price for NVIDIA GeForce GTX 1080 Ti", ...


def _ceil(x: float) -> int:
    return math.ceil(x - 1e-9)  # 60/60 must stay 1, not become 2 through float error


def _price_assumption(price: GpuPrice) -> str:
    return f"{price.gpu} at ${price.hourly_usd_low:g}-{price.hourly_usd_high:g} per GPU-hour (prices as of {price.as_of})"


def _sum(estimates: list[Estimate], unit: str) -> Estimate:
    assumptions = list(dict.fromkeys(a for e in estimates for a in e.assumptions))
    return Estimate(low=sum(e.low for e in estimates), high=sum(e.high for e in estimates), unit=unit, assumptions=assumptions)


def monthly_cost(gpu_count: float | Estimate, price: GpuPrice) -> Estimate:
    """gpu_count x hourly_usd_low..high x HOURS_PER_MONTH. With an Estimate count, low x low and high x high."""
    if isinstance(gpu_count, Estimate):
        low_count, high_count = gpu_count.low, gpu_count.high
    else:
        low_count = high_count = gpu_count
    return Estimate(
        low=low_count * price.hourly_usd_low * HOURS_PER_MONTH,
        high=high_count * price.hourly_usd_high * HOURS_PER_MONTH,
        unit="USD/month",
        assumptions=[_price_assumption(price), f"{HOURS_PER_MONTH} hours per month, running all the time"],
    )


def tokens_per_hour(records: list[CallRecord]) -> float | None:
    """Input + output tokens over the records' own time span (first started_at to last
    completed_at, else started_at), per hour. None when no record has token counts or the span
    is shorter than MIN_TOKEN_SPAN. Records with missing counts are skipped, not zeroed."""
    counted = [
        r
        for r in records
        if r.usage is not None and r.usage.input_tokens is not None and r.usage.output_tokens is not None
    ]
    if not counted:
        return None
    start = min(r.started_at for r in counted)
    end = max(r.completed_at or r.started_at for r in counted)
    if end - start < MIN_TOKEN_SPAN:
        return None
    tokens = sum(r.usage.input_tokens + r.usage.output_tokens for r in counted)
    return tokens / ((end - start).total_seconds() / 3600)


def consolidate(assessment: GpuAssessment, target_pct: tuple[float, float] = TARGET_UTILIZATION_PCT) -> Consolidation | None:
    """GPUs needed: ceil(sum of per-GPU mean utilization / target), for the high and low
    target, floored by the GPUs memory needs. None without utilization readings."""
    means = [u.utilization_pct.mean for u in assessment.usage if u.utilization_pct is not None]
    if not means:
        return None
    busy = sum(means)
    low_target, high_target = target_pct
    by_util_low, by_util_high = max(1, _ceil(busy / high_target)), max(1, _ceil(busy / low_target))

    peaks = [u.memory_used_mib.max for u in assessment.usage if u.memory_used_mib is not None]
    totals = [u.memory_total_mib for u in assessment.usage if u.memory_total_mib]
    by_memory = max(1, _ceil(sum(peaks) / min(totals))) if peaks and totals else 1

    return Consolidation(
        current_gpus=assessment.gpu_count,
        needed_gpus=Estimate(
            low=max(by_util_low, by_memory),
            high=max(by_util_high, by_memory),
            unit="GPUs",
            assumptions=[
                REPRESENTATIVE,
                f"GPUs run at {low_target:g}-{high_target:g}% utilization after consolidating",
                "models can be moved between GPUs freely (tensor-parallel layouts may need more)",
            ],
        ),
        limited_by="memory" if by_memory >= by_util_high else "utilization",
    )


def size_instances(assessment: GpuAssessment, consolidation: Consolidation, prices: PriceTable) -> Sizing | None:
    """The cheapest (by hourly_usd_low) priced GPU whose memory fits the largest per-GPU peak
    memory use. None when no priced GPU fits."""
    peaks = [u.memory_used_mib.max for u in assessment.usage if u.memory_used_mib is not None]
    if not peaks:
        return None
    fits = [p for p in prices.prices if p.memory_gib * 1024 >= max(peaks)]
    if not fits:
        return None
    best = min(fits, key=lambda p: p.hourly_usd_low)
    count = consolidation.needed_gpus.model_copy(
        update={"assumptions": [*consolidation.needed_gpus.assumptions, f"one {best.gpu} does the work of one current GPU"]}
    )
    return Sizing(gpu=best.gpu, count=count, monthly_cost=monthly_cost(count, best))


def cost_per_million_tokens(hourly_cost: Estimate, tokens_hour: float) -> Estimate:
    """hourly_cost / tokens_hour x 1e6, low and high."""
    return Estimate(
        low=hourly_cost.low / tokens_hour * 1e6,
        high=hourly_cost.high / tokens_hour * 1e6,
        unit="USD per 1M tokens",
        assumptions=[*hourly_cost.assumptions, "the tokens in the recorder logs were all served by these GPUs"],
    )


def savings(current: Estimate, proposed: Estimate) -> tuple[Estimate, Estimate]:
    """(monthly, annual). Monthly low = current.low - proposed.high, high = current.high - proposed.low."""
    assumptions = list(dict.fromkeys([*current.assumptions, *proposed.assumptions]))
    low, high = current.low - proposed.high, current.high - proposed.low
    monthly = Estimate(low=low, high=high, unit="USD/month", assumptions=assumptions)
    annual = Estimate(low=low * 12, high=high * 12, unit="USD/year", assumptions=assumptions)
    return monthly, annual


def estimate_costs(
    assessment: GpuAssessment,
    profile: GpuProfile,
    prices: PriceTable,
    records: list[CallRecord] | None = None,
) -> CostEstimates:
    """All of the above. Price the current GPUs by matching snapshot GPU names (match_gpu);
    mixed GPU types are priced per GPU. Fill `missing` for every figure left None."""
    out = CostEstimates()

    matched = [(g, match_gpu(g.name, prices)) for g in profile.snapshot.gpus]
    unpriced = sorted({g.name for g, p in matched if p is None})
    if unpriced:
        out.missing.append(f"no price in gpu_prices.yaml for {', '.join(unpriced)}, so current cost and savings weren't estimated")
    else:
        out.current_monthly_cost = _sum([monthly_cost(1, p) for _, p in matched], "USD/month")

    out.consolidation = consolidate(assessment)
    if out.consolidation is None:
        out.missing.append("no GPU utilization readings, so consolidation wasn't estimated")
    else:
        out.sizing = size_instances(assessment, out.consolidation, prices)
        if out.sizing is None:
            out.missing.append("no priced GPU type has enough memory for the measured peak, so no instance size was suggested")

    if out.current_monthly_cost is not None and out.sizing is not None:
        out.monthly_savings, out.annual_savings = savings(out.current_monthly_cost, out.sizing.monthly_cost)

    if out.current_monthly_cost is not None:
        rate = tokens_per_hour(records) if records else None
        if rate is None:
            out.missing.append("no token counts spanning at least 10 minutes in the recorder logs, so cost per million tokens wasn't estimated")
        else:
            m = out.current_monthly_cost
            hourly = m.model_copy(update={"low": m.low / HOURS_PER_MONTH, "high": m.high / HOURS_PER_MONTH, "unit": "USD/hour"})
            out.cost_per_million_tokens = cost_per_million_tokens(hourly, rate)

    return out
