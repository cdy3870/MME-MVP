"""The GPU report: GPU-5 to GPU-9 in one object, for report.json / the UI to render.

    assessment       GPU-5, measured
    recommendations  GPU-6, estimated gains
    costs            GPU-7, estimated
    checklist        GPU-9
    untested_notice  GPU-8, shown above the recommendations and the checklist

Rendering (labels on screen, the notice) belongs to the report writer / frontend. This
module only assembles; nothing here is computed twice.
"""

from __future__ import annotations

from pydantic import BaseModel

from migration_eval.gpu.assessment import GpuAssessment, assess
from migration_eval.gpu.checklist import ChecklistItem, build_checklist
from migration_eval.gpu.costs import CostEstimates, estimate_costs
from migration_eval.gpu.labels import UNTESTED_NOTICE
from migration_eval.gpu.prices import PriceTable
from migration_eval.gpu.profile import GpuProfile
from migration_eval.gpu.recommendations import Recommendation, recommend
from migration_eval.models import CallRecord


class GpuReport(BaseModel):
    hostname: str
    assessment: GpuAssessment
    recommendations: list[Recommendation]
    costs: CostEstimates | None  # None without a price table
    checklist: list[ChecklistItem]
    untested_notice: str = UNTESTED_NOTICE


def build_gpu_report(
    profile: GpuProfile,
    prices: PriceTable | None = None,
    records: list[CallRecord] | None = None,
) -> GpuReport:
    """assess -> recommend -> estimate_costs (when prices) -> build_checklist."""
    assessment = assess(profile)
    recommendations = recommend(assessment, profile)
    costs = estimate_costs(assessment, profile, prices, records) if prices is not None else None
    return GpuReport(
        hostname=profile.hostname,
        assessment=assessment,
        recommendations=recommendations,
        costs=costs,
        checklist=build_checklist(recommendations, costs),
    )
