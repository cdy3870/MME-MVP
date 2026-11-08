"""GPU-1 samples: parsing `--query-gpu` CSV and the sampling loop, with no GPU."""

from __future__ import annotations

import pytest

from migration_eval.gpu.samples import QUERY_FIELDS, collect, parse_csv, query_args

# Remove once the module is implemented; strict=True makes a passing test fail as a reminder.
pytestmark = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="GPU-1 skeleton")

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


def test_collect_takes_duration_over_interval_ticks():
    now = [0.0]
    calls = []

    def run(args):
        calls.append(args)
        return TICK

    def sleep(seconds):
        now[0] += seconds

    samples = collect(run, interval_s=5, duration_s=30, clock=lambda: now[0], sleep=sleep)
    assert len(calls) == 6
    assert len(samples) == 12  # 6 ticks x 2 GPUs


def test_collect_rejects_interval_longer_than_duration():
    with pytest.raises(ValueError):
        collect(lambda args: TICK, interval_s=10, duration_s=5)
