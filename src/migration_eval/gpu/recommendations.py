"""GPU-6: ranked configuration recommendations (Deliverable 2).

Rules over the GPU-5 assessment and the GPU-4 serving settings. Each rule looks at the
findings (and the profile where it needs more) and returns a Recommendation or None. `recommend`
runs every rule and keeps the top MAX_RECOMMENDATIONS by expected gain.

Areas the ticket asks for: serving settings (batch size, memory, parallelism, precision,
caching), GPU allocation, NUMA and CPU affinity, persistence mode and clocks, and MPS or MIG
when several processes share GPUs. vLLM-specific advice is GPU-14, not here.

Every expected gain is an Estimate (GPU-8): a range with its assumptions. The gains in
TYPICAL_GAIN_PCT are starting guesses, to tune on real captures; say so in the assumptions.
Recommendations are written as experiments ("try X, then compare Y"), since none is tested.

Each recommendation also gets the checklist stage GPU-9 orders by:
    fix       something is wrong now (throttling, placement, affinity)
    runtime   a serving or driver setting to try in staging
    capacity  more, fewer or different GPUs
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from migration_eval.gpu.assessment import Finding, GpuAssessment
from migration_eval.gpu.labels import Estimate
from migration_eval.gpu.profile import GpuProfile

MAX_RECOMMENDATIONS = 5

Level = Literal["low", "medium", "high"]
Stage = Literal["fix", "runtime", "capacity"]
Area = Literal["serving", "allocation", "affinity", "clocks", "sharing"]

# Expected throughput gain, % (low, high). GUESSES to tune; they go into the assumptions.
TYPICAL_GAIN_PCT: dict[str, tuple[float, float]] = {
    "fix_thermal": (5, 20),
    "fix_power": (5, 15),
    "pin_numa": (2, 10),
    "rebalance": (10, 30),
    "raise_batch": (10, 40),
    "lower_memory_fraction": (0, 5),  # mainly stability (fewer OOMs), little throughput
    "persistence_mode": (0, 2),  # mainly first-request latency
    "share_gpus": (10, 50),  # MPS / MIG for several small processes on one GPU
}
EFFORT_ORDER = {"low": 0, "medium": 1, "high": 2}

LOW_UTILIZATION_PCT = 50.0  # mean utilization below this leaves room for bigger batches
BATCH_HEADROOM_PCT = 25.0  # ... when at least this much memory was free at peak on every GPU
SHARED_GPU_UTILIZATION_PCT = 40.0  # more GPU processes than GPUs, on GPUs this lightly used -> MPS/MIG
MIG_GPUS = ("A100", "A30", "H100", "H200", "H20", "B200")


class Recommendation(BaseModel):
    id: str  # stable rule id, e.g. "pin_numa"; keys TYPICAL_GAIN_PCT
    title: str  # one line: "Pin the serving process to the GPU's NUMA node"
    area: Area
    stage: Stage
    action: str  # what to change, concretely (command or flag where we can give one)
    experiment: str  # how to check it worked: what to measure before and after
    expected_gain: Estimate  # unit "%" throughput, basis "estimated"
    effort: Level
    risk: Level
    gpus: list[int] = []  # GPUs it applies to; empty = all
    based_on: list[str] = []  # Finding kinds (or "serving") that triggered it


Rule = Callable[[GpuAssessment, GpuProfile], Recommendation | None]


def _findings(assessment: GpuAssessment, kind: str) -> list[Finding]:
    return [f for f in assessment.findings if f.kind == kind]


def _gpus(findings: list[Finding]) -> list[int]:
    return sorted({g for f in findings for g in f.gpus})


def _gpu_list(gpus: list[int]) -> str:
    return ", ".join(str(g) for g in gpus)


def _mean_utilization(assessment: GpuAssessment) -> float | None:
    means = [u.utilization_pct.mean for u in assessment.usage if u.utilization_pct is not None]
    return sum(means) / len(means) if means else None


def gain(rule_id: str, *extra_assumptions: str) -> Estimate:
    """The Estimate for a rule's TYPICAL_GAIN_PCT, with the 'typical gain, untested' assumption."""
    low, high = TYPICAL_GAIN_PCT[rule_id]
    return Estimate(
        low=low,
        high=high,
        unit="%",
        assumptions=["typical throughput gain for this change, not measured on your machine", *extra_assumptions],
    )


# --- rules: one per recommendation -------------------------------------------------------------


def fix_thermal(assessment: GpuAssessment, profile: GpuProfile) -> Recommendation | None:
    """`thermal` or `hw_slowdown` findings -> check cooling/airflow on those GPUs. Stage fix."""
    found = _findings(assessment, "thermal") + _findings(assessment, "hw_slowdown")
    if not found:
        return None
    gpus = _gpus(found)
    return Recommendation(
        id="fix_thermal",
        title=f"Fix the cooling on GPU {_gpu_list(gpus)}",
        area="clocks",
        stage="fix",
        action="Check fan speeds, airflow and inlet temperature for these GPUs "
        "(`nvidia-smi -q -d TEMPERATURE,PERFORMANCE`), and ask the hosting provider if it is a cloud machine. "
        "Hardware slowdown can also point to power delivery.",
        experiment="Re-run `migration-eval gpu capture` under the same load and compare the slowdown share and the SM clock.",
        expected_gain=gain("fix_thermal"),
        effort="medium",
        risk="low",
        gpus=gpus,
        based_on=sorted({f.kind for f in found}),
    )


def fix_power(assessment: GpuAssessment, profile: GpuProfile) -> Recommendation | None:
    """`power_limit` finding -> check the power limit (`nvidia-smi -q -d POWER`) against the
    card's default/max. If it was lowered, raising it is the fix; if it's already at max, say
    the card is power-bound and that's expected under heavy load. Stage fix."""
    found = _findings(assessment, "power_limit")
    if not found:
        return None
    gpus = _gpus(found)
    return Recommendation(
        id="fix_power",
        title=f"Check the power limit on GPU {_gpu_list(gpus)}",
        area="clocks",
        stage="fix",
        action="Compare 'Current Power Limit' with 'Default Power Limit' and 'Max Power Limit' in "
        "`nvidia-smi -q -d POWER`. If it was lowered, restore it (`sudo nvidia-smi -i <gpu> -pl <watts>`). "
        "If it is already at the maximum, the GPU is power-bound under this load, which is expected.",
        experiment="Re-capture under the same load and compare p95 power draw, the power-cap share and throughput.",
        expected_gain=gain("fix_power", "only applies if the limit was lowered below the default"),
        effort="low",
        risk="medium",
        gpus=gpus,
        based_on=["power_limit"],
    )


def pin_numa(assessment: GpuAssessment, profile: GpuProfile) -> Recommendation | None:
    """`numa_affinity` or `cross_socket` findings -> run each serving process with
    `numactl --cpunodebind=N --membind=N` for its GPU's NUMA node (from topology.gpus_by_numa_node).
    Stage fix."""
    found = _findings(assessment, "numa_affinity") + _findings(assessment, "cross_socket")
    if not found:
        return None
    nodes = assessment.topology.gpus_by_numa_node
    layout = "; ".join(f"node {node}: GPU {_gpu_list(gpus)}" for node, gpus in sorted(nodes.items()))
    return Recommendation(
        id="pin_numa",
        title="Keep each serving process on its GPU's NUMA node",
        area="affinity",
        stage="fix",
        action="Start each serving process with `numactl --cpunodebind=<node> --membind=<node>` for the node its GPUs "
        f"are attached to{f' ({layout})' if layout else ''}, and keep tensor-parallel groups within one socket where possible.",
        experiment="Compare throughput and p95 latency before and after pinning, at the same request rate.",
        expected_gain=gain("pin_numa"),
        effort="low",
        risk="low",
        gpus=_gpus(found),
        based_on=sorted({f.kind for f in found}),
    )


def rebalance(assessment: GpuAssessment, profile: GpuProfile) -> Recommendation | None:
    """`imbalance` finding -> spread load: check the load balancer / replica-to-GPU mapping, or
    the tensor-parallel layout. Name the busiest and idlest GPU. Stage runtime."""
    imb = assessment.imbalance
    if imb is None or not _findings(assessment, "imbalance"):
        return None
    return Recommendation(
        id="rebalance",
        title=f"Spread load more evenly (GPU {imb.busiest} is {imb.spread_pct:.0f} points busier than GPU {imb.idlest})",
        area="allocation",
        stage="runtime",
        action="Check how requests reach each GPU: the load balancer's weights, how many replicas sit on each GPU, "
        "and CUDA_VISIBLE_DEVICES for each server. Move or add a replica on the idle GPU.",
        experiment="Re-capture at the same load: the utilization spread should drop and total throughput rise.",
        expected_gain=gain("rebalance", f"utilization spread of {imb.spread_pct:.0f} points measured during the capture"),
        effort="medium",
        risk="low",
        gpus=[imb.busiest, imb.idlest],
        based_on=["imbalance"],
    )


def raise_batch(assessment: GpuAssessment, profile: GpuProfile) -> Recommendation | None:
    """Low utilization with memory headroom -> try a larger max batch size. Uses the serving
    batch size when known (profile.serving); otherwise says the current value is unknown. Skip
    when memory headroom is low, or when the capture looked idle (low utilization there means
    no traffic, not small batches). Stage runtime."""
    if _findings(assessment, "idle_capture"):
        return None
    mean = _mean_utilization(assessment)
    headrooms = [u.memory_headroom_pct for u in assessment.usage if u.memory_headroom_pct is not None]
    if mean is None or mean >= LOW_UTILIZATION_PCT or not headrooms or min(headrooms) < BATCH_HEADROOM_PCT:
        return None
    batches = sorted({s.batch_size for s in profile.serving if s.batch_size is not None})
    current = f"currently {', '.join(map(str, batches))}" if batches else "current value unknown"
    return Recommendation(
        id="raise_batch",
        title=f"Try a larger maximum batch size ({current})",
        area="serving",
        stage="runtime",
        action=f"GPUs averaged {mean:.0f}% busy with at least {min(headrooms):.0f}% of memory free. Raise the engine's "
        "maximum batch size (vLLM: --max-num-seqs) in steps, for example 1.5x then 2x.",
        experiment="At each step, load-test with the production request mix; keep the largest step whose p95 latency stays on target.",
        expected_gain=gain("raise_batch", f"mean utilization of {mean:.0f}% measured during the capture"),
        effort="low",
        risk="medium",
        based_on=["serving"] if batches else [],
    )


def lower_memory_fraction(assessment: GpuAssessment, profile: GpuProfile) -> Recommendation | None:
    """`memory_headroom` warning on a non-vLLM engine (vLLM pre-allocates, so its finding is
    only "info") -> lower the engine's memory fraction or batch size to avoid OOMs. Stage runtime."""
    found = [f for f in _findings(assessment, "memory_headroom") if f.severity != "info"]
    if not found:
        return None
    gpus = _gpus(found)
    return Recommendation(
        id="lower_memory_fraction",
        title=f"Leave more memory free on GPU {_gpu_list(gpus)}",
        area="serving",
        stage="runtime",
        action="Memory peaked within 10% of the total. Lower the engine's GPU memory fraction or maximum batch size "
        "a step, or move a model off these GPUs, so bursts don't run out of memory.",
        experiment="Watch for out-of-memory errors and restarts over a busy day, and check throughput didn't drop.",
        expected_gain=gain("lower_memory_fraction", "the gain is mainly stability, not speed"),
        effort="low",
        risk="low",
        gpus=gpus,
        based_on=["memory_headroom"],
    )


def persistence_mode(assessment: GpuAssessment, profile: GpuProfile) -> Recommendation | None:
    """Persistence mode off -> `nvidia-smi -pm 1` (Linux). Stage runtime, low effort, low risk.
    None when it's on everywhere or wasn't reported (Windows reports N/A)."""
    off = [g.index for g in profile.snapshot.gpus if g.persistence_mode is False]
    if not off:
        return None
    return Recommendation(
        id="persistence_mode",
        title=f"Turn on persistence mode (off on GPU {_gpu_list(off)})",
        area="clocks",
        stage="runtime",
        action="Run `sudo nvidia-smi -pm 1`, or enable the nvidia-persistenced service so it survives reboots. "
        "It keeps the driver initialized between jobs.",
        experiment="Compare the latency of the first request after an idle period.",
        expected_gain=gain("persistence_mode", "the gain is mainly first-request latency"),
        effort="low",
        risk="low",
        gpus=off,
    )


def share_gpus(assessment: GpuAssessment, profile: GpuProfile) -> Recommendation | None:
    """Several serving processes sharing lightly used GPUs -> MPS, or MIG on GPUs that have
    it. Stage capacity, medium effort.

    The profile doesn't record which GPU each process used, so the trigger is: more GPU
    processes (profile.process_samples with uses_gpu) than GPUs, and mean utilization below
    SHARED_GPU_UTILIZATION_PCT."""
    gpu_pids = {p.pid for p in profile.process_samples if p.uses_gpu}
    mean = _mean_utilization(assessment)
    if assessment.gpu_count == 0 or len(gpu_pids) <= assessment.gpu_count:
        return None
    if mean is None or mean >= SHARED_GPU_UTILIZATION_PCT:
        return None
    mig = any(any(m in g.name for m in MIG_GPUS) for g in profile.snapshot.gpus)
    how = "MIG partitions (these GPUs support MIG) or MPS" if mig else "MPS (these GPUs don't support MIG)"
    return Recommendation(
        id="share_gpus",
        title="Share GPUs between processes more efficiently",
        area="sharing",
        stage="capacity",
        action=f"{len(gpu_pids)} GPU processes ran on {assessment.gpu_count} GPU(s) averaging {mean:.0f}% busy. "
        f"Use {how} so they run side by side instead of time-slicing, then consider fewer GPUs.",
        experiment="Run the same processes under MPS or MIG in staging and compare throughput and p95 latency per process.",
        expected_gain=gain("share_gpus"),
        effort="medium",
        risk="medium",
    )


RULES: tuple[Rule, ...] = (
    fix_thermal,
    fix_power,
    pin_numa,
    rebalance,
    raise_batch,
    lower_memory_fraction,
    persistence_mode,
    share_gpus,
)


def rank(recommendations: list[Recommendation], limit: int = MAX_RECOMMENDATIONS) -> list[Recommendation]:
    """Highest expected_gain.midpoint first; ties go to lower effort, then lower risk. Keep `limit`."""
    ordered = sorted(
        recommendations,
        key=lambda r: (-r.expected_gain.midpoint, EFFORT_ORDER[r.effort], EFFORT_ORDER[r.risk]),
    )
    return ordered[:limit]


def recommend(assessment: GpuAssessment, profile: GpuProfile, rules: tuple[Rule, ...] = RULES) -> list[Recommendation]:
    """Run every rule, drop the Nones, rank. A rule that raises is a bug: let it raise."""
    return rank([r for rule in rules if (r := rule(assessment, profile)) is not None])
