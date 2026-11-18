"""GPU-9: the ordered checklist at the end of the GPU report.

The partner's execution plan, in three stages, in this order:

    fix       throttling, placement, affinity: something is wrong now
    runtime   serving and driver settings, to try in staging
    capacity  more, fewer or different GPUs (consolidation, sizing, MPS/MIG)

Items come from the recommendations (GPU-6, which already carry a stage) and from the cost
estimates (GPU-7: a consolidation with savings becomes a capacity item). Within a stage,
highest expected gain first. Each item shows its expected gain and risk. No re-test follows
in the MVP; the report puts UNTESTED_NOTICE (GPU-8) above the list.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from migration_eval.gpu.costs import CostEstimates
from migration_eval.gpu.labels import Estimate
from migration_eval.gpu.recommendations import Level, Recommendation, Stage

STAGE_ORDER: tuple[Stage, ...] = ("fix", "runtime", "capacity")
UNIT_ORDER = {"%": 0}  # within a stage: throughput gains before money; other units after


class ChecklistItem(BaseModel):
    step: int  # 1-based position in the list
    stage: Stage
    title: str
    action: str
    expected_gain: Estimate  # "%" throughput, or "USD/month" for a capacity saving
    risk: Level
    source: Literal["recommendation", "cost"]
    source_id: str  # Recommendation.id, or "consolidation"


def from_recommendation(rec: Recommendation) -> ChecklistItem:
    """One item from a recommendation (step filled in by build_checklist)."""
    return ChecklistItem(
        step=0,
        stage=rec.stage,
        title=rec.title,
        action=rec.action,
        expected_gain=rec.expected_gain,
        risk=rec.risk,
        source="recommendation",
        source_id=rec.id,
    )


def from_costs(costs: CostEstimates) -> ChecklistItem | None:
    """A capacity item when consolidation needs fewer GPUs than today and monthly_savings.high > 0;
    gain is the monthly savings, risk "medium". None otherwise."""
    c, saved = costs.consolidation, costs.monthly_savings
    if c is None or saved is None or c.needed_gpus.high >= c.current_gpus or saved.high <= 0:
        return None
    target = costs.sizing.gpu if costs.sizing is not None else "GPUs"
    low, high = int(c.needed_gpus.low), int(c.needed_gpus.high)
    count = str(low) if low == high else f"{low}-{high}"
    return ChecklistItem(
        step=0,
        stage="capacity",
        title=f"Consolidate from {c.current_gpus} to {count} {target}",
        action="After the fixes and runtime changes above, move the load onto fewer GPUs in staging and check "
        "latency and throughput hold at peak traffic before releasing the rest.",
        expected_gain=saved,
        risk="medium",
        source="cost",
        source_id="consolidation",
    )


def build_checklist(recommendations: list[Recommendation], costs: CostEstimates | None = None) -> list[ChecklistItem]:
    """Order by STAGE_ORDER, then expected_gain.midpoint descending within a stage (compare
    gains only within the same unit; "%" items before "USD/month" items in a stage). Number
    the steps 1..n."""
    items = [from_recommendation(r) for r in recommendations]
    if costs is not None and (item := from_costs(costs)) is not None:
        items.append(item)
    items.sort(
        key=lambda i: (
            STAGE_ORDER.index(i.stage),
            UNIT_ORDER.get(i.expected_gain.unit, 1),
            -i.expected_gain.midpoint,
        )
    )
    return [item.model_copy(update={"step": n}) for n, item in enumerate(items, start=1)]
