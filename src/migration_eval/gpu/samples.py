"""Timed samples: `nvidia-smi --query-gpu=<fields> --format=csv,noheader,nounits`.

Default: every 5 seconds for 10 minutes (`--interval`, `--duration`), so 120 rows per GPU.

Run one query per tick rather than `nvidia-smi -l`: each tick is then a complete, separate
reading, a slow tick can't merge rows, and tests can drive the loop with a fake clock.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel

from migration_eval.gpu.nvidia_smi import Runner

DEFAULT_INTERVAL_S = 5.0
DEFAULT_DURATION_S = 600.0

# Order matters: it is the column order of the CSV.
QUERY_FIELDS = (
    "timestamp",
    "index",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
    "clocks_throttle_reasons.active",  # renamed clocks_event_reasons.active in newer drivers
)


class Sample(BaseModel):
    """One GPU at one tick. Fields the driver reports as "[N/A]" or "[Not Supported]" are None."""

    taken_at: datetime
    index: int
    utilization_gpu_pct: float | None = None
    utilization_memory_pct: float | None = None
    memory_used_mib: float | None = None
    memory_total_mib: float | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None
    temperature_c: float | None = None
    sm_clock_mhz: float | None = None
    mem_clock_mhz: float | None = None
    throttle_reasons: int | None = None  # bitmask, e.g. 0x20 = SW thermal slowdown


def query_args(fields: tuple[str, ...] = QUERY_FIELDS) -> list[str]:
    """The arguments for one sample: `--query-gpu=a,b,c --format=csv,noheader,nounits`."""
    raise NotImplementedError


def parse_csv(csv_text: str, fields: tuple[str, ...] = QUERY_FIELDS) -> list[Sample]:
    """Parse one tick's output: one CSV row per GPU.

    Use the `csv` module. Strip whitespace around values. Map "[N/A]" / "[Not Supported]" to
    None. `throttle_reasons` is hex ("0x0000000000000004"). Raise ValueError on a row with the
    wrong number of columns.
    """
    raise NotImplementedError


def resolve_fields(run: Runner) -> tuple[str, ...]:
    """Return QUERY_FIELDS with the throttle field name this driver accepts.

    Try one query with `clocks_throttle_reasons.active`; if the driver rejects it, use
    `clocks_event_reasons.active`. Decide once, before sampling.
    """
    raise NotImplementedError


def collect(
    run: Runner,
    interval_s: float = DEFAULT_INTERVAL_S,
    duration_s: float = DEFAULT_DURATION_S,
    *,
    fields: tuple[str, ...] = QUERY_FIELDS,
    clock: Callable[[], float] | None = None,  # default time.monotonic
    sleep: Callable[[float], None] | None = None,  # default time.sleep
    on_tick: Callable[[int, int], None] | None = None,  # (tick, total_ticks), for a progress line
) -> list[Sample]:
    """Sample every `interval_s` until `duration_s` has passed and return all rows.

    Schedule ticks against the start time (start + n * interval), not "sleep interval after each
    query", so the run doesn't drift. On Ctrl-C, stop and return what was collected so far; the
    caller decides whether that's enough to write. Validate interval_s > 0 and
    duration_s >= interval_s.
    """
    raise NotImplementedError
