"""GPU-5: the GPU performance assessment."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from migration_eval.gpu.assessment import (
    assess,
    compute_imbalance,
    summarize,
    summarize_throttling,
    usage_by_gpu,
)
from migration_eval.gpu.profile import CaptureSettings, GpuProfile
from migration_eval.gpu.samples import Sample
from migration_eval.gpu.snapshot import GpuInfo, Snapshot, Topology


T0 = datetime(2026, 11, 7, 20, 0, tzinfo=timezone.utc)


def gpu(index: int) -> GpuInfo:
    return GpuInfo(index=index, uuid=f"GPU-{index}", name="NVIDIA H100", pci_bus_id=f"0:{index}", memory_total_mib=81559, power_limit_w=700)


def sample(index: int, i: int, util: float, mem: float = 40000, power: float = 400, throttle: int = 0) -> Sample:
    return Sample(
        taken_at=T0 + timedelta(seconds=5 * i), index=index, utilization_gpu_pct=util,
        memory_used_mib=mem, memory_total_mib=81559, power_draw_w=power, power_limit_w=700,
        temperature_c=70, sm_clock_mhz=1980, throttle_reasons=throttle,
    )


SNAPSHOT = Snapshot(
    driver_version="550.54.15",
    gpus=[gpu(0), gpu(1)],
    topology=Topology(
        links={"GPU0": {"GPU1": "SYS"}, "GPU1": {"GPU0": "SYS"}},
        cpu_affinity={"GPU0": "0-31", "GPU1": "32-63"},
        numa_affinity={"GPU0": "0", "GPU1": "1"},
    ),
    raw_xml="",
    raw_topo="",
)


def profile(samples: list[Sample]) -> GpuProfile:
    return GpuProfile(
        hostname="gpu01",
        settings=CaptureSettings(source="capture", interval_s=5, duration_s=50, started_at=T0, ended_at=T0 + timedelta(seconds=50)),
        snapshot=SNAPSHOT,
        samples=samples,
    )


def test_summarize():
    stat = summarize([1.0, 2.0, 3.0, 4.0])
    assert (stat.mean, stat.min, stat.max, stat.samples, stat.basis) == (2.5, 1.0, 4.0, 4, "measured")
    assert summarize([]) is None


def test_imbalance_between_gpus():
    usage = usage_by_gpu([sample(0, i, 90) for i in range(10)] + [sample(1, i, 30) for i in range(10)], SNAPSHOT)
    imbalance = compute_imbalance(usage)
    assert imbalance.spread_pct == 60 and (imbalance.busiest, imbalance.idlest) == (0, 1)


def test_idle_bit_is_not_throttling():
    (t,) = summarize_throttling([sample(0, i, 0, throttle=0x1) for i in range(10)])
    assert t.fraction_by_reason["gpu_idle"] == 1.0
    assert t.slowdown_fraction == 0.0


def test_power_cap_counts_as_throttling():
    rows = [sample(0, i, 99, throttle=0x4 if i < 5 else 0) for i in range(10)]
    (t,) = summarize_throttling(rows)
    assert t.slowdown_fraction == 0.5 and t.power_fraction == 0.5 and t.thermal_fraction == 0.0


def test_assess_finds_imbalance_power_and_cross_socket():
    rows = [sample(0, i, 95, mem=78000, power=690, throttle=0x4) for i in range(10)]
    rows += [sample(1, i, 20) for i in range(10)]
    result = assess(profile(rows))
    kinds = {f.kind for f in result.findings}
    assert {"imbalance", "power_limit", "memory_headroom", "cross_socket"} <= kinds
    assert result.topology.interconnect == "pcie"
    assert result.topology.cross_socket_pairs == [(0, 1)]
    assert all(f.basis == "measured" for f in result.findings)


def test_single_gpu_has_no_imbalance():
    result = assess(profile([sample(0, i, 50) for i in range(5)]))
    assert result.imbalance is None
