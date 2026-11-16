"""GPU-2: the idle warning."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from migration_eval.gpu.idle import check_idle, format_idle_warning
from migration_eval.gpu.samples import Sample


T0 = datetime(2026, 11, 7, 20, 0, tzinfo=timezone.utc)


def samples(utilization_by_gpu: dict[int, list[float | None]]) -> list[Sample]:
    return [
        Sample(taken_at=T0 + timedelta(seconds=2 * i), index=gpu, utilization_gpu_pct=u)
        for gpu, values in utilization_by_gpu.items()
        for i, u in enumerate(values)
    ]


def test_idle_desktop_warns():
    # The real idle capture on the GTX 1080 Ti: 8% once, then 0%.
    warning = check_idle(samples({0: [8.0] + [0.0] * 14}))
    assert warning is not None
    assert warning.idle_gpus == [0]
    assert "normal traffic" in format_idle_warning(warning)


def test_loaded_capture_does_not_warn():
    # The real load capture: ~14 s of model loading, then ~50% while generating.
    assert check_idle(samples({0: [2, 2, 4, 5, 8, 10, 11, 14] + [50.0] * 22})) is None


def test_one_busy_gpu_is_not_an_idle_capture():
    assert check_idle(samples({0: [0.0] * 10, 1: [90.0] * 10})) is None


def test_missing_utilization_is_ignored():
    assert check_idle(samples({0: [None] * 5 + [0.0] * 10})) is not None


def test_no_samples():
    assert check_idle([]) is None
