"""GPU-1 samples: parsing `--query-gpu` CSV and the sampling loop, with no GPU."""

from __future__ import annotations

import pytest

from migration_eval.gpu.nvidia_smi import NvidiaSmiError
from migration_eval.gpu.samples import (
    QUERY_FIELDS,
    THROTTLE_FIELD_NEW,
    THROTTLE_FIELD_OLD,
    collect,
    parse_csv,
    query_args,
    resolve_fields,
)

# Two GPUs, one tick. GPU1 reports power as [N/A] and is thermally throttled (0x20).
TICK = (
    "2026/11/07 12:00:00.000, 0, 87, 41, 61234, 81559, 512.30, 700.00, 71, 1980, 2619, 0x0000000000000000\n"
    "2026/11/07 12:00:00.000, 1, 12, 3, 1024, 81559, [N/A], 700.00, 88, 1410, 2619, 0x0000000000000020\n"
)


def test_query_args_requests_machine_readable_csv():
    args = query_args()
    assert args[0] == "--query-gpu=" + ",".join(QUERY_FIELDS)
    assert "--format=csv,noheader,nounits" in args


def test_parse_csv_one_row_per_gpu():
    samples = parse_csv(TICK)
    assert [s.index for s in samples] == [0, 1]
    assert samples[0].utilization_gpu_pct == 87
    assert samples[0].power_draw_w == pytest.approx(512.3)


def test_parse_csv_maps_na_to_none_and_reads_hex_throttle_mask():
    gpu1 = parse_csv(TICK)[1]
    assert gpu1.power_draw_w is None
    assert gpu1.throttle_reasons == 0x20


def test_parse_csv_rejects_wrong_column_count():
    with pytest.raises(ValueError):
        parse_csv("2026/11/07 12:00:00.000, 0, 87\n")


def test_parse_csv_timestamp_is_timezone_aware():
    assert parse_csv(TICK)[0].taken_at.tzinfo is not None


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_collect_takes_duration_over_interval_ticks():
    clock = FakeClock()
    calls = []

    def run(args):
        calls.append(args)
        return TICK

    result = collect(run, interval_s=5, duration_s=30, clock=clock, sleep=clock.sleep)
    assert len(calls) == 6
    assert len(result.samples) == 12  # 6 ticks x 2 GPUs
    assert (result.ticks_planned, result.ticks_taken, result.interrupted) == (6, 6, False)


def test_collect_schedules_against_start_so_slow_ticks_do_not_drift():
    clock = FakeClock()

    def slow_run(args):
        clock.now += 2  # each query takes 2s
        return TICK

    collect(slow_run, interval_s=5, duration_s=15, clock=clock, sleep=clock.sleep)
    assert clock.sleeps == [3, 3]  # not [5, 5]


def test_collect_skips_failed_ticks():
    outputs = iter([TICK, NvidiaSmiError("boom"), "garbage", TICK])

    def run(args):
        out = next(outputs)
        if isinstance(out, Exception):
            raise out
        return out

    clock = FakeClock()
    result = collect(run, interval_s=1, duration_s=4, clock=clock, sleep=clock.sleep)
    assert (result.ticks_taken, result.ticks_failed) == (2, 2)
    assert len(result.samples) == 4


def test_collect_ctrl_c_keeps_what_was_collected():
    calls = []

    def run(args):
        calls.append(args)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return TICK

    clock = FakeClock()
    result = collect(run, interval_s=1, duration_s=10, clock=clock, sleep=clock.sleep)
    assert result.interrupted
    assert result.ticks_taken == 2 and len(result.samples) == 4


def test_collect_rejects_interval_longer_than_duration():
    with pytest.raises(ValueError):
        collect(lambda args: TICK, interval_s=10, duration_s=5)


def test_resolve_fields_falls_back_to_renamed_throttle_field():
    def run(args):
        if THROTTLE_FIELD_OLD in args[0]:
            raise NvidiaSmiError('Field "clocks_throttle_reasons.active" is not a valid field to query.')
        return TICK

    fields = resolve_fields(run)
    assert THROTTLE_FIELD_NEW in fields and THROTTLE_FIELD_OLD not in fields
    assert parse_csv(TICK, fields)[1].throttle_reasons == 0x20


def test_resolve_fields_keeps_old_name_when_accepted():
    assert resolve_fields(lambda args: TICK) == QUERY_FIELDS


def test_cpu_sampler_interval_rounds_up_to_whole_seconds():
    # mpstat/pidstat only take whole seconds (cpu_samples.py).
    from migration_eval.gpu.cpu_samples import mpstat_args, whole_seconds

    assert (whole_seconds(0.5), whole_seconds(5), whole_seconds(5.2)) == (1, 5, 6)
    assert mpstat_args(2.5, 10)[-2:] == ["3", "10"]
