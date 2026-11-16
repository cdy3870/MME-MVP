"""GPU-2: warn when the capture looks idle.

The readings are only useful under normal traffic. When utilization stays near zero for
most of the capture, tell the partner to re-run during normal traffic. It's a warning:
the profile is still written and `export` still includes it.

Where it shows up:
  - `gpu capture` prints it after sampling, before "Wrote <file>".
  - It is stored in the profile (`GpuProfile.idle_warning`), so `export` can repeat it in
    its summary and the report (GPU-5) can say the readings may not be representative.

Checked against a real capture on a GTX 1080 Ti: idle desktop gave 0% on 14 of 15 samples
(must warn); a load run that spent its first 14 s loading a model and then ran at ~50% must
not warn.
"""

from __future__ import annotations

from pydantic import BaseModel

from migration_eval.gpu.samples import Sample

IDLE_UTILIZATION_PCT = 5.0  # a sample at or below this counts as idle
IDLE_FRACTION = 0.9  # a GPU is idle when at least this share of its samples are


class IdleWarning(BaseModel):
    idle_gpus: list[int]  # indices of GPUs that were idle
    gpu_count: int
    idle_fraction_by_gpu: dict[int, float]  # share of samples at or below the threshold
    mean_utilization_pct: float  # across all GPUs and samples
    message: str


def check_idle(
    samples: list[Sample],
    threshold_pct: float = IDLE_UTILIZATION_PCT,
    idle_fraction: float = IDLE_FRACTION,
) -> IdleWarning | None:
    """Return a warning when every GPU was idle, else None.

    Per GPU: the share of its samples with utilization_gpu_pct <= threshold_pct (samples
    where utilization is None don't count either way). Warn only when every GPU is idle;
    a machine where some GPUs work and some don't is an imbalance for GPU-5, not an idle
    capture. Return None for no samples.

    Message: say how idle it was, and that the readings only help when taken under normal
    traffic, so re-run `gpu capture` while the service is serving.
    """
    if len(samples) == 0:
        return None

    counted: dict[int, int] = {}  # GPU -> samples with a utilization reading
    idle: dict[int, int] = {}  # GPU -> of those, samples at or below the threshold
    readings: list[float] = []
    for sample in samples:
        if sample.utilization_gpu_pct is None:
            continue
        counted[sample.index] = counted.get(sample.index, 0) + 1
        idle.setdefault(sample.index, 0)
        if sample.utilization_gpu_pct <= threshold_pct:
            idle[sample.index] += 1
        readings.append(sample.utilization_gpu_pct)

    if not counted:
        return None  # no utilization readings at all: nothing to judge

    idle_fraction_by_gpu = {gpu: idle[gpu] / counted[gpu] for gpu in sorted(counted)}
    if any(fraction < idle_fraction for fraction in idle_fraction_by_gpu.values()):
        return None

    mean = sum(readings) / len(readings)
    least_idle = min(idle_fraction_by_gpu.values())
    gpus = "the GPU" if len(counted) == 1 else f"all {len(counted)} GPUs"
    message = (
        f"{gpus} sat at or below {threshold_pct:g}% utilization for at least {least_idle:.0%} of the "
        f"capture (mean {mean:.1f}%). The readings only help when taken under normal traffic, so "
        "re-run `gpu capture` while the service is serving."
    )
    return IdleWarning(
        idle_gpus=sorted(counted),
        gpu_count=len(counted),
        idle_fraction_by_gpu=idle_fraction_by_gpu,
        mean_utilization_pct=mean,
        message=message[0].upper() + message[1:],
    )


def format_idle_warning(warning: IdleWarning) -> str:
    """The text `gpu capture` and `export` print, starting with "Warning:"."""
    return f"Warning: {warning.message}"
