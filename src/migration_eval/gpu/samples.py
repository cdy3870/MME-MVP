"""Timed samples: `nvidia-smi --query-gpu=<fields> --format=csv,noheader,nounits`.

Default: every 5 seconds for 10 minutes (`--interval`, `--duration`), so 120 rows per GPU.

Run one query per tick rather than `nvidia-smi -l`: each tick is then a complete, separate
reading, a slow tick can't merge rows, and tests can drive the loop with a fake clock.
"""

from __future__ import annotations

import csv
import io
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel

from migration_eval.gpu.nvidia_smi import NvidiaSmiError, Runner

DEFAULT_INTERVAL_S = 5.0
DEFAULT_DURATION_S = 600.0

THROTTLE_FIELD_OLD = "clocks_throttle_reasons.active"
THROTTLE_FIELD_NEW = "clocks_event_reasons.active"

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
    THROTTLE_FIELD_OLD,  # renamed clocks_event_reasons.active in newer drivers
)

# nvidia-smi field -> Sample attribute.
FIELD_ATTRS = {
    "timestamp": "taken_at",
    "index": "index",
    "utilization.gpu": "utilization_gpu_pct",
    "utilization.memory": "utilization_memory_pct",
    "memory.used": "memory_used_mib",
    "memory.total": "memory_total_mib",
    "power.draw": "power_draw_w",
    "power.limit": "power_limit_w",
    "temperature.gpu": "temperature_c",
    "clocks.sm": "sm_clock_mhz",
    "clocks.mem": "mem_clock_mhz",
    THROTTLE_FIELD_OLD: "throttle_reasons",
    THROTTLE_FIELD_NEW: "throttle_reasons",
}

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S.%f"  # e.g. "2026/11/07 12:00:00.000", local time


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


@dataclass
class SampleRun:
    """What `collect` gathered, and whether the run went to plan."""

    samples: list[Sample] = field(default_factory=list)
    ticks_planned: int = 0
    ticks_taken: int = 0  # ticks that returned rows
    ticks_failed: int = 0  # ticks where nvidia-smi failed or its output didn't parse
    interrupted: bool = False  # stopped with Ctrl-C
    last_error: str | None = None


def query_args(fields: tuple[str, ...] = QUERY_FIELDS) -> list[str]:
    """The arguments for one sample: `--query-gpu=a,b,c --format=csv,noheader,nounits`."""
    return [f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]


def _missing(value: str) -> bool:
    # "[N/A]", "[Not Supported]", "[Unknown Error]", "[GPU requires reset]", ...
    return not value or value == "N/A" or (value.startswith("[") and value.endswith("]"))


def _convert(attr: str, value: str) -> object:
    if _missing(value):
        return None
    if attr == "taken_at":
        # nvidia-smi prints local time with no offset; attach the machine's offset.
        return datetime.strptime(value, TIMESTAMP_FORMAT).astimezone()
    if attr == "index":
        return int(value)
    if attr == "throttle_reasons":
        return int(value, 16)
    return float(value)


def parse_csv(csv_text: str, fields: tuple[str, ...] = QUERY_FIELDS) -> list[Sample]:
    """Parse one tick's output: one CSV row per GPU.

    Raise ValueError on an unknown field, a row with the wrong number of columns, or a value
    that doesn't parse.
    """
    unknown = [f for f in fields if f not in FIELD_ATTRS]
    if unknown:
        raise ValueError(f"unknown query fields: {', '.join(unknown)}")
    attrs = [FIELD_ATTRS[f] for f in fields]

    samples: list[Sample] = []
    for row in csv.reader(io.StringIO(csv_text), skipinitialspace=True):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != len(fields):
            raise ValueError(f"expected {len(fields)} columns, got {len(row)}: {','.join(row)}")
        values = {attr: _convert(attr, cell.strip()) for attr, cell in zip(attrs, row)}
        samples.append(Sample.model_validate(values))
    return samples


def resolve_fields(run: Runner) -> tuple[str, ...]:
    """Return QUERY_FIELDS with the throttle field name this driver accepts.

    Try one query with `clocks_throttle_reasons.active`; if the driver rejects it, use
    `clocks_event_reasons.active`. If that fails too, the NvidiaSmiError propagates: some
    other field is the problem, and its message names it.
    """
    try:
        run(query_args(QUERY_FIELDS))
        return QUERY_FIELDS
    except NvidiaSmiError:
        renamed = tuple(THROTTLE_FIELD_NEW if f == THROTTLE_FIELD_OLD else f for f in QUERY_FIELDS)
        run(query_args(renamed))
        return renamed


def planned_ticks(interval_s: float, duration_s: float) -> int:
    """How many samples a run takes: duration / interval, rounded down (600s / 5s = 120)."""
    if interval_s <= 0:
        raise ValueError("interval must be greater than 0")
    if duration_s < interval_s:
        raise ValueError("duration must be at least one interval")
    return math.floor(duration_s / interval_s + 1e-9)  # 0.3 / 0.1 is 2.9999...


def collect(
    run: Runner,
    interval_s: float = DEFAULT_INTERVAL_S,
    duration_s: float = DEFAULT_DURATION_S,
    *,
    fields: tuple[str, ...] = QUERY_FIELDS,
    clock: Callable[[], float] | None = None,  # default time.monotonic
    sleep: Callable[[float], None] | None = None,  # default time.sleep
    on_tick: Callable[[int, int], None] | None = None,  # (tick, total_ticks), for a progress line
) -> SampleRun:
    """Sample every `interval_s` until `duration_s` has passed.

    Tick n is scheduled at start + n * interval, not "interval after the previous query", so
    the run doesn't drift. A tick that fails (nvidia-smi error or unparseable output) is
    counted and skipped rather than ending a 10-minute capture. Ctrl-C stops the run and
    returns what was collected so far, with `interrupted=True`.
    """
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    result = SampleRun(ticks_planned=planned_ticks(interval_s, duration_s))
    args = query_args(fields)

    start = clock()
    try:
        for tick in range(result.ticks_planned):
            wait = start + tick * interval_s - clock()
            if wait > 0:
                sleep(wait)
            try:
                result.samples.extend(parse_csv(run(args), fields))
                result.ticks_taken += 1
            except (NvidiaSmiError, ValueError) as exc:
                result.ticks_failed += 1
                result.last_error = str(exc)
            if on_tick is not None:
                on_tick(tick + 1, result.ticks_planned)
    except KeyboardInterrupt:
        result.interrupted = True
    return result
