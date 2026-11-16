"""GPU-5: the GPU performance assessment (Deliverable 1).

Utilization and imbalance across GPUs, memory and power headroom, heat and clock throttling,
and topology (NVLink or PCIe, CPU and NUMA affinity). Every number here is measured from the
partner's own readings, and is labelled so (GPU-8). The recommendations (GPU-6) and the
checklist (GPU-9) read the `findings`.

This module computes; it doesn't render. The report writer turns a GpuAssessment into
report.json / report.html sections. `migration-eval gpu assess <profile>` prints it as JSON
for development.

Throttling: nvidia-smi's clocks event reasons mix real slowdowns with bits that only say why
clocks are where they are. Bit 0x1 ("GPU idle") is set whenever there's no work; counting
it would report every idle GPU as throttled. Only SLOWDOWN_BITS count.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Literal

import numpy as np
from pydantic import BaseModel

from migration_eval.gpu.profile import GpuProfile
from migration_eval.gpu.samples import Sample
from migration_eval.gpu.snapshot import Snapshot


# nvidia-smi clocks_event_reasons bits.
THROTTLE_REASONS = {
    0x1: "gpu_idle",
    0x2: "applications_clocks_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}
SLOWDOWN_BITS = 0x4 | 0x8 | 0x20 | 0x40 | 0x80  # what counts as throttling
THERMAL_BITS = 0x20 | 0x40
POWER_BITS = 0x4 | 0x80

# Thresholds for findings. Starting points, to tune on real captures.
IMBALANCE_SPREAD_PCT = 20.0  # busiest minus idlest mean utilization
LOW_MEMORY_HEADROOM_PCT = 10.0  # peak memory use above 90% of total
POWER_NEAR_LIMIT_PCT = 95.0  # p95 power draw at or above 95% of the limit
THROTTLED_FRACTION = 0.05  # slowdown active in at least 5% of samples
CROSS_SOCKET_LINKS = ("SYS",)  # topo -m: traffic crosses the CPU interconnect between sockets


class Stat(BaseModel):
    """Summary of one reading over the capture. Always measured (GPU-8)."""

    basis: Literal["measured"] = "measured"
    mean: float
    p50: float
    p95: float
    min: float
    max: float
    samples: int


class GpuUsage(BaseModel):
    index: int
    name: str
    utilization_pct: Stat | None = None  # None when the driver didn't report it
    memory_used_mib: Stat | None = None
    power_draw_w: Stat | None = None
    temperature_c: Stat | None = None
    sm_clock_mhz: Stat | None = None
    # Limits, and how much room was left against them (measured, GPU-8).
    memory_total_mib: float | None = None  # snapshot, else the samples
    power_limit_w: float | None = None  # samples (limit in force), else the snapshot
    memory_headroom_pct: float | None = None  # 100 * (total - memory_used_mib.max) / total
    power_headroom_pct: float | None = None  # 100 * (limit - power_draw_w.p95) / limit


class Imbalance(BaseModel):
    basis: Literal["measured"] = "measured"
    spread_pct: float  # busiest mean utilization minus idlest
    coefficient_of_variation: float  # stdev / mean of per-GPU mean utilization
    busiest: int
    idlest: int


class Throttling(BaseModel):
    basis: Literal["measured"] = "measured"
    index: int
    fraction_by_reason: dict[str, float]  # reason name -> share of samples it was active
    slowdown_fraction: float  # share of samples with any SLOWDOWN_BITS bit set
    thermal_fraction: float
    power_fraction: float
    sm_clock_p50_when_slowed_mhz: float | None = None  # how deep the slowdown went


class TopologySummary(BaseModel):
    interconnect: Literal["nvlink", "pcie", "mixed", "single_gpu", "unknown"]
    link_counts: dict[str, int]  # topo -m link type ("NV12", "PIX", "SYS", ...) -> GPU pairs
    cross_socket_pairs: list[tuple[int, int]]  # GPU pairs whose traffic crosses sockets
    gpus_by_numa_node: dict[str, list[int]]
    active_nvlinks_by_gpu: dict[int, int]  # from `nvlink -s`; empty when not collected
    pcie: dict[int, str]  # GPU index -> "Gen4 x16" (max link; current link not yet collected)
    serving_cpus_outside_gpu_numa: list[int]  # serving-process PIDs that ran on CPUs not local to their GPUs


class Finding(BaseModel):
    """One observation for GPU-6 and GPU-9 to act on."""

    kind: Literal["imbalance", "memory_headroom", "power_limit", "thermal", "hw_slowdown", "cross_socket", "numa_affinity", "idle_capture", "missing_data"]
    severity: Literal["info", "warning", "critical"]
    gpus: list[int]
    message: str
    basis: Literal["measured"] = "measured"


class GpuAssessment(BaseModel):
    gpu_count: int
    usage: list[GpuUsage]
    imbalance: Imbalance | None  # None for a single GPU
    throttling: list[Throttling]
    topology: TopologySummary
    findings: list[Finding]
    missing: list[str]  # readings we couldn't take (commands missing, fields N/A)


def summarize(values: list[float | None]) -> Stat | None:
    """Stat over the values, ignoring None; None when there are none."""
    values = [v for v in values if v is not None]
    if len(values) == 0:
        return None
    v = np.array(values, dtype=float)
    return Stat(mean=float(v.mean()),
                p50=float(np.percentile(v, 50)),
                p95=float(np.percentile(v, 95)),
                min=float(v.min()),
                max=float(v.max()),
                samples=len(values))


def _last(values: list[float | None]) -> float | None:
    """The last value that isn't None."""
    for v in reversed(values):
        if v is not None:
            return v
    return None


def usage_by_gpu(samples: list[Sample], snapshot: Snapshot) -> list[GpuUsage]:
    """Group samples by GPU index and summarize each reading. Names come from the snapshot."""

    groups = {}
    gpu_usage = []

    for sample in samples:
        groups.setdefault(sample.index, []).append(sample)

    for gpu in snapshot.gpus:
        rows = groups.get(gpu.index, [])
        memory_used_stats = summarize([row.memory_used_mib for row in rows])
        power_draw_stats = summarize([row.power_draw_w for row in rows])

        # Limits: total memory from the snapshot, else the samples; power limit in force from
        # the samples (an admin may have lowered it), else the snapshot.
        memory_total = gpu.memory_total_mib or _last([row.memory_total_mib for row in rows])
        power_limit = _last([row.power_limit_w for row in rows]) or gpu.power_limit_w

        memory_headroom_pct = None
        if memory_used_stats is not None and memory_total:
            memory_headroom_pct = 100 * (memory_total - memory_used_stats.max) / memory_total

        power_headroom_pct = None
        if power_draw_stats is not None and power_limit:
            power_headroom_pct = 100 * (power_limit - power_draw_stats.p95) / power_limit

        gpu_usage.append(GpuUsage(index=gpu.index,
                                    name=gpu.name,
                                    utilization_pct=summarize([row.utilization_gpu_pct for row in rows]),
                                    memory_used_mib=memory_used_stats,
                                    power_draw_w=power_draw_stats,
                                    temperature_c=summarize([row.temperature_c for row in rows]),
                                    sm_clock_mhz=summarize([row.sm_clock_mhz for row in rows]),
                                    memory_total_mib=memory_total,
                                    power_limit_w=power_limit,
                                    memory_headroom_pct=memory_headroom_pct,
                                    power_headroom_pct=power_headroom_pct))

    return gpu_usage


def compute_imbalance(usage: list[GpuUsage]) -> Imbalance | None:
    """From per-GPU mean utilization. None for fewer than two GPUs with utilization."""

    means = {u.index: u.utilization_pct.mean for u in usage if u.utilization_pct is not None}

    if len(means) < 2:
        return None

    spread_pct = max(means.values()) - min(means.values())

    vals = np.array(list(means.values()))
    cv = float(vals.std() / vals.mean()) if vals.mean() > 0 else 0.0

    busiest = max(means, key=means.get)
    idlest = min(means, key=means.get)

    imbalance = Imbalance(spread_pct=spread_pct, coefficient_of_variation=cv, busiest=busiest, idlest=idlest)

    return imbalance


def summarize_throttling(samples: list[Sample]) -> list[Throttling]:
    """Per GPU: share of samples each THROTTLE_REASONS bit was set, and the share with any
    SLOWDOWN_BITS, THERMAL_BITS or POWER_BITS bit set. Samples where throttle_reasons is None
    don't count. gpu_idle is reported in fraction_by_reason but never counts as a slowdown."""

    groups = {}
    for sample in samples:
        if sample.throttle_reasons is not None:
            groups.setdefault(sample.index, []).append(sample)

    throttles = []
    for index in sorted(groups):
        rows = groups[index]
        n = len(rows)

        reason_counts = {name: 0 for name in THROTTLE_REASONS.values()}
        slowdown_count = 0
        thermal_count = 0
        power_count = 0
        sm_clock_when_slowed = []

        for row in rows:
            mask = row.throttle_reasons
            for bit, name in THROTTLE_REASONS.items():
                if mask & bit:
                    reason_counts[name] += 1
            if mask & SLOWDOWN_BITS:
                slowdown_count += 1
                sm_clock_when_slowed.append(row.sm_clock_mhz)
            if mask & THERMAL_BITS:
                thermal_count += 1
            if mask & POWER_BITS:
                power_count += 1

        stat = summarize(sm_clock_when_slowed)
        throttles.append(Throttling(index=index,
                                    fraction_by_reason={k: c / n for k, c in reason_counts.items() if c},
                                    slowdown_fraction=slowdown_count / n,
                                    thermal_fraction=thermal_count / n,
                                    power_fraction=power_count / n,
                                    sm_clock_p50_when_slowed_mhz=stat.p50 if stat else None))

    return throttles


def _gpu_index(name: str) -> int | None:
    """"GPU3" -> 3; None for anything else ("NIC0", ...)."""
    if name.startswith("GPU") and name[3:].isdigit():
        return int(name[3:])
    return None


def _serving_cpus_outside_gpu_numa(profile: GpuProfile, gpus_by_numa_node: dict[str, list[int]]) -> list[int]:
    """PIDs of GPU processes that ran on a core outside every NUMA node that has a GPU.

    The profile doesn't record which GPU each process uses, so this only catches a process
    running on a socket with no GPUs at all. Mapping PIDs to GPUs (compute-apps gpu_uuid)
    would allow the per-GPU check the TopologySummary field describes.
    """
    if profile.host is None or profile.host.numa is None or not gpus_by_numa_node:
        return []
    gpu_local_cpus = {
        cpu
        for node in profile.host.numa.nodes
        if str(node.id) in gpus_by_numa_node
        for cpu in node.cpus
    }
    if not gpu_local_cpus:
        return []
    return sorted({
        p.pid
        for p in profile.process_samples
        if p.uses_gpu and p.cpu is not None and p.cpu not in gpu_local_cpus
    })


def summarize_topology(profile: GpuProfile) -> TopologySummary:
    """From snapshot.topology, snapshot.nvlink, host.numa and the serving processes' CPUs.

    interconnect: "nvlink" when every GPU pair is NV#, "pcie" when none is, else "mixed".
    gpus_by_numa_node: from topo -m's NUMA affinity column. serving_cpus_outside_gpu_numa:
    compare ProcessCpuSample.cpu for GPU processes against the CPUs of their GPU's NUMA node
    (host.numa). Degrade to "unknown" / empty when the inputs weren't collected.
    """

    snapshot = profile.snapshot

    pcie = {
        g.index: f"Gen{g.pcie_gen_max} x{g.pcie_width_max}"
        for g in snapshot.gpus
        if g.pcie_gen_max is not None and g.pcie_width_max is not None
    }

    active_nvlinks_by_gpu = {}
    for gpu, links in (snapshot.nvlink or {}).items():
        active_nvlinks_by_gpu[gpu] = sum(1 for link in links if link.active)

    topology = snapshot.topology

    if topology is None:
        return TopologySummary(interconnect="single_gpu" if len(snapshot.gpus) < 2 else "unknown",
                                link_counts={},
                                cross_socket_pairs=[],
                                gpus_by_numa_node={},
                                active_nvlinks_by_gpu=active_nvlinks_by_gpu,
                                pcie=pcie,
                                serving_cpus_outside_gpu_numa=[],
                            )

    # Each GPU pair once (index < index_2); topo -m lists both directions.
    methods = []
    cross_socket_link_pairs = []
    for name, connections in topology.links.items():
        index = _gpu_index(name)
        if index is None:
            continue
        for name_2, method in connections.items():
            index_2 = _gpu_index(name_2)
            if index_2 is None or index_2 <= index:
                continue  # a NIC column, or a pair already counted
            methods.append(method)
            if method in CROSS_SOCKET_LINKS:
                cross_socket_link_pairs.append((index, index_2))
    cross_socket_link_pairs.sort()

    link_counts = dict(Counter(methods))

    nv = [method.startswith("NV") for method in methods]
    if not methods:
        interconnect = "single_gpu"
    elif all(nv):
        interconnect = "nvlink"
    elif not any(nv):
        interconnect = "pcie"
    else:
        interconnect = "mixed"

    gpus_by_numa_node = defaultdict(list)
    for name, node in topology.numa_affinity.items():
        index = _gpu_index(name)
        if index is not None and node != "N/A":
            gpus_by_numa_node[node].append(index)
    gpus_by_numa_node = {node: sorted(gpus) for node, gpus in gpus_by_numa_node.items()}

    return TopologySummary(interconnect=interconnect,
                            link_counts=link_counts,
                            cross_socket_pairs=cross_socket_link_pairs,
                            gpus_by_numa_node=gpus_by_numa_node,
                            active_nvlinks_by_gpu=active_nvlinks_by_gpu,
                            pcie=pcie,
                            serving_cpus_outside_gpu_numa=_serving_cpus_outside_gpu_numa(profile, gpus_by_numa_node),
                        )


SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _gpu_list(indexes: list[int]) -> str:
    return ", ".join(str(i) for i in indexes)


def build_findings(assessment: GpuAssessment, profile: GpuProfile) -> list[Finding]:
    """Turn the numbers into findings using the thresholds at the top of this module.

    Also: idle_capture when the profile carries an idle warning (GPU-2), and missing_data for
    each important input that wasn't collected (topology, NUMA, CPU samples), so the report
    says what it couldn't assess. Most severe first.
    """

    findings = []

    if profile.snapshot.topology is None and len(profile.snapshot.gpus) > 1:
        findings.append(Finding(kind="missing_data", severity="info", gpus=[],
                                message="`nvidia-smi topo -m` wasn't collected, so the GPU interconnect and NUMA affinity weren't assessed."))
    if profile.host is None or profile.host.numa is None:
        findings.append(Finding(kind="missing_data", severity="info", gpus=[],
                                message="The NUMA layout (`numactl --hardware`) wasn't collected, so CPU placement of serving processes wasn't assessed."))
    if not profile.cpu_samples:
        findings.append(Finding(kind="missing_data", severity="info", gpus=[],
                                message="CPU samples (`mpstat`) weren't collected, so host CPU load wasn't assessed."))

    imbalance = assessment.imbalance
    if imbalance is not None and imbalance.spread_pct >= IMBALANCE_SPREAD_PCT:
        findings.append(Finding(kind="imbalance", severity="warning", gpus=[imbalance.busiest, imbalance.idlest],
                                message=f"GPU utilization is uneven: GPU {imbalance.busiest} averaged {imbalance.spread_pct:.0f} "
                                        f"points higher than GPU {imbalance.idlest}."))

    # vLLM reserves most GPU memory for its KV cache up front, so low headroom is expected there.
    uses_vllm = any(s.engine.lower() == "vllm" for s in profile.serving)

    lows = []
    power_limited = []

    for u in assessment.usage:
        if u.memory_headroom_pct is not None and u.memory_headroom_pct < LOW_MEMORY_HEADROOM_PCT:
            lows.append(u)
        if u.power_headroom_pct is not None and u.power_headroom_pct <= 100 - POWER_NEAR_LIMIT_PCT:
            power_limited.append(u)

    if lows:
        detail = "; ".join(f"GPU {u.index} peaked at {u.memory_used_mib.max:.0f} of {u.memory_total_mib:.0f} MiB "
                           f"({u.memory_headroom_pct:.1f}% free)" for u in lows)
        note = " Expected with vLLM, which reserves memory for its KV cache." if uses_vllm else ""
        findings.append(Finding(kind="memory_headroom", severity="info" if uses_vllm else "warning",
                                gpus=[u.index for u in lows], message=f"Little GPU memory left: {detail}.{note}"))

    power_fraction = {t.index: t.power_fraction for t in assessment.throttling}
    if power_limited:
        detail = "; ".join(f"GPU {u.index} p95 {u.power_draw_w.p95:.0f} W of a {u.power_limit_w:.0f} W limit, "
                           f"power-capped in {power_fraction.get(u.index, 0):.0%} of samples" for u in power_limited)
        findings.append(Finding(kind="power_limit", severity="warning", gpus=[u.index for u in power_limited],
                                message=f"Running at the power limit: {detail}."))

    thermal_throttled = []
    hw_throttled = []
    for t in assessment.throttling:
        if t.thermal_fraction >= THROTTLED_FRACTION:
            thermal_throttled.append(t)
        # Only the hardware bits; slowdown_fraction also includes thermal and the power cap.
        hw_fraction = t.fraction_by_reason.get("hw_slowdown", 0) + t.fraction_by_reason.get("hw_power_brake_slowdown", 0)
        if hw_fraction >= THROTTLED_FRACTION:
            hw_throttled.append((t, hw_fraction))

    if thermal_throttled:
        detail = "; ".join(f"GPU {t.index} in {t.thermal_fraction:.0%} of samples" for t in thermal_throttled)
        findings.append(Finding(kind="thermal", severity="warning", gpus=[t.index for t in thermal_throttled],
                                message=f"Clocks slowed for heat: {detail}."))

    if hw_throttled:
        detail = "; ".join(f"GPU {t.index} in {f:.0%} of samples" for t, f in hw_throttled)
        findings.append(Finding(kind="hw_slowdown", severity="critical", gpus=[t.index for t, _ in hw_throttled],
                                message=f"Hardware slowdown (cooling or power delivery): {detail}."))

    pairs = assessment.topology.cross_socket_pairs
    if pairs:
        gpus = sorted({i for pair in pairs for i in pair})
        findings.append(Finding(kind="cross_socket", severity="warning", gpus=gpus,
                                message=f"{len(pairs)} GPU pair(s) talk across CPU sockets (topo -m SYS), the slowest path, "
                                        f"involving GPU {_gpu_list(gpus)}."))

    outside = assessment.topology.serving_cpus_outside_gpu_numa
    if outside:
        findings.append(Finding(kind="numa_affinity", severity="warning", gpus=[],
                                message=f"GPU process(es) {_gpu_list(outside)} ran on CPU cores outside the GPUs' NUMA nodes."))

    if profile.idle_warning:
        findings.append(Finding(kind="idle_capture", severity="warning", gpus=profile.idle_warning.idle_gpus,
                                message=profile.idle_warning.message))

    return sorted(findings, key=lambda f: SEVERITY_ORDER[f.severity])


def _missing(profile: GpuProfile, usage: list[GpuUsage]) -> list[str]:
    """Readings we couldn't take, so the report can say what it couldn't see."""
    missing = []
    if profile.snapshot.topology is None:
        missing.append("topology (nvidia-smi topo -m)")
    if profile.snapshot.nvlink is None:
        missing.append("NVLink status (nvidia-smi nvlink -s)")
    if profile.host is None:
        missing.append("host CPU/NUMA (lscpu, numactl)")
    elif profile.host.numa is None:
        missing.append("NUMA (numactl --hardware)")
    if not profile.cpu_samples:
        missing.append("CPU samples (mpstat)")
    if not profile.process_samples:
        missing.append("process CPU samples (pidstat)")
    if not profile.serving:
        missing.append("serving settings")
    for field in ("utilization_pct", "memory_used_mib", "power_draw_w", "temperature_c", "sm_clock_mhz"):
        gpus = [u.index for u in usage if getattr(u, field) is None]
        if gpus:
            missing.append(f"{field} for GPU {_gpu_list(gpus)}")
    return missing


def assess(profile: GpuProfile) -> GpuAssessment:
    """The whole assessment for one profile."""
    usage = usage_by_gpu(profile.samples, profile.snapshot)
    assessment = GpuAssessment(
        gpu_count=len(profile.snapshot.gpus),
        usage=usage,
        imbalance=compute_imbalance(usage),
        throttling=summarize_throttling(profile.samples),
        topology=summarize_topology(profile),
        findings=[],
        missing=_missing(profile, usage),
    )
    assessment.findings = build_findings(assessment, profile)
    return assessment
