"""GPU-6: ranked configuration recommendations."""

from __future__ import annotations

import pytest

from migration_eval.gpu.assessment import assess
from migration_eval.gpu.labels import Estimate
from migration_eval.gpu.recommendations import (
    MAX_RECOMMENDATIONS,
    Recommendation,
    fix_thermal,
    pin_numa,
    rank,
    rebalance,
    recommend,
)

# tests/ has no __init__.py files, so pytest puts this directory on sys.path (prepend mode).
from test_gpu_assessment import profile, sample



def rec(id: str, low: float, high: float, effort="low", risk="low", stage="runtime") -> Recommendation:
    return Recommendation(
        id=id, title=id, area="serving", stage=stage, action="a", experiment="e",
        expected_gain=Estimate(low=low, high=high, unit="%"), effort=effort, risk=risk,
    )  # fmt: skip


def imbalanced_cross_socket():
    # SNAPSHOT in test_gpu_assessment puts GPU0 and GPU1 on different sockets (SYS).
    p = profile([sample(0, i, 90) for i in range(10)] + [sample(1, i, 10) for i in range(10)])
    return assess(p), p


def test_rebalance_names_busiest_and_idlest():
    assessment, p = imbalanced_cross_socket()
    r = rebalance(assessment, p)
    assert r is not None and r.stage == "runtime" and "imbalance" in r.based_on
    assert r.expected_gain.basis == "estimated"


def test_pin_numa_on_cross_socket_is_a_fix():
    assessment, p = imbalanced_cross_socket()
    r = pin_numa(assessment, p)
    assert r is not None and r.stage == "fix" and "numactl" in r.action


def test_fix_thermal_only_when_throttled():
    calm = profile([sample(0, i, 80) for i in range(10)] + [sample(1, i, 80) for i in range(10)])
    assert fix_thermal(assess(calm), calm) is None
    hot = profile([sample(0, i, 80, throttle=0x40) for i in range(10)] + [sample(1, i, 80) for i in range(10)])
    r = fix_thermal(assess(hot), hot)
    assert r is not None and r.gpus == [0]


def test_rank_by_gain_then_effort():
    ranked = rank([rec("a", 0, 10), rec("b", 10, 30), rec("c", 10, 30, effort="high")])
    assert [r.id for r in ranked] == ["b", "c", "a"]


def test_recommend_keeps_at_most_five():
    assessment, p = imbalanced_cross_socket()
    recs = recommend(assessment, p)
    assert 0 < len(recs) <= MAX_RECOMMENDATIONS


def test_raise_batch_on_light_load_with_free_memory():
    p = profile([sample(g, i, 20, mem=20000) for g in (0, 1) for i in range(10)])
    r = recommend(assess(p), p)
    assert "raise_batch" in [x.id for x in r]


def test_no_raise_batch_when_memory_is_tight():
    p = profile([sample(g, i, 20, mem=78000) for g in (0, 1) for i in range(10)])
    assert "raise_batch" not in [x.id for x in recommend(assess(p), p)]


def test_persistence_mode_off_is_recommended():
    p = profile([sample(g, i, 80) for g in (0, 1) for i in range(10)])
    p.snapshot.gpus[1].persistence_mode = False
    r = [x for x in recommend(assess(p), p) if x.id == "persistence_mode"]
    assert r and r[0].gpus == [1]
