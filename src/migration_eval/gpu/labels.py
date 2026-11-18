"""GPU-8: every number in the GPU report is labelled measured or estimated.

Measured: read from the partner's own capture (GPU-5's Stat, Imbalance, Throttling, ...).
These already carry `basis: "measured"`.

Estimated: our calculation on top of the readings, under stated assumptions: expected gains
(GPU-6), GPU counts, costs and savings (GPU-7). Use `Estimate` for every such number, so it
is a range with its assumptions attached, never a bare float.

Nothing is applied and re-measured in the MVP, so the report also carries UNTESTED_NOTICE
next to the recommendations and the checklist.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

Basis = Literal["measured", "estimated"]

UNTESTED_NOTICE = (
    "Recommendations and estimates are based on one capture and have not been tested on your "
    "machine. Treat each one as an experiment: apply it in staging and measure before relying on it. "
    "Re-testing comes in v1."
)


class Estimate(BaseModel):
    """A number we estimated, shown as a range."""

    basis: Literal["estimated"] = "estimated"
    low: float
    high: float
    unit: str  # "%", "GPUs", "USD/month", "USD per 1M tokens", ...
    assumptions: list[str] = []  # shown with the number, e.g. "prices as of 2026-11-01"

    @model_validator(mode="after")
    def _ordered(self) -> Estimate:
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) is above high ({self.high})")
        return self

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2
