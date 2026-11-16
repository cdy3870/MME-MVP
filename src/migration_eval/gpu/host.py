"""The host half of the GPU-1 snapshot: CPU, NUMA and InfiniBand.

    lscpu -J              JSON. CPU model, sockets, cores, threads, NUMA node count.
    numactl --hardware    Text. Per-node CPUs and memory, and the node distance matrix.
    ibstat                Text, only when the machine has InfiniBand. Per-port state and rate.
    ibv_devinfo -v        Text, the same. Kept raw for now; ibstat has what the report needs.

numactl and ibstat have no machine-readable output, so, like `topo -m`, parse them by the
labels they print and keep the raw text (in `commands`) for re-parsing later.

The GPU-side additions (`nvidia-smi nvlink -s`) live in snapshot.py with the other nvidia-smi
commands.
"""

from __future__ import annotations

import json
import os
import re

from pydantic import BaseModel

from migration_eval.gpu.commands import CommandResult, CommandRunner

LSCPU_ARGS = ("lscpu", "-J")
NUMACTL_ARGS = ("numactl", "--hardware")
IBSTAT_ARGS = ("ibstat",)
IBV_DEVINFO_ARGS = ("ibv_devinfo", "-v")
INFINIBAND_SYSFS = "/sys/class/infiniband"


class CpuInfo(BaseModel):
    model_name: str | None = None
    architecture: str | None = None
    cpus: int | None = None  # logical CPUs
    sockets: int | None = None
    cores_per_socket: int | None = None
    threads_per_core: int | None = None
    numa_nodes: int | None = None


class NumaNode(BaseModel):
    id: int
    cpus: list[int]
    memory_total_mib: int | None = None
    memory_free_mib: int | None = None


class Numa(BaseModel):
    nodes: list[NumaNode]
    distances: list[list[int]]  # distances[i][j]; 10 = local


class InfinibandPort(BaseModel):
    device: str  # e.g. "mlx5_0"
    port: int
    state: str | None = None  # "Active", "Down", ...
    physical_state: str | None = None  # "LinkUp", "Polling", ...
    rate_gbps: float | None = None
    link_layer: str | None = None  # "InfiniBand" or "Ethernet" (RoCE)


class HostSnapshot(BaseModel):
    cpu: CpuInfo | None = None  # None when lscpu is missing or didn't parse
    numa: Numa | None = None
    infiniband: list[InfinibandPort] | None = None  # None = no InfiniBand found or not readable
    commands: list[CommandResult] = []  # every command run, with raw output and status


# lscpu field (trailing colon stripped) -> CpuInfo attribute.
LSCPU_FIELDS = {
    "Model name": "model_name",
    "Architecture": "architecture",
    "CPU(s)": "cpus",
    "Socket(s)": "sockets",
    "Core(s) per socket": "cores_per_socket",
    "Thread(s) per core": "threads_per_core",
    "NUMA node(s)": "numa_nodes",
}
_LSCPU_INT_ATTRS = {"cpus", "sockets", "cores_per_socket", "threads_per_core", "numa_nodes"}


def _flatten_lscpu(entries: list) -> list[dict]:
    flat = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        flat.append(entry)
        flat.extend(_flatten_lscpu(entry.get("children") or []))
    return flat


def parse_lscpu_json(text: str) -> CpuInfo:
    """Parse `lscpu -J`: {"lscpu": [{"field": "CPU(s):", "data": "16"}, ...]}.

    Newer util-linux nests entries under "children"; flatten them first. Match on the field
    name with the trailing colon stripped. Raise ValueError if it isn't that JSON shape.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not JSON: {exc}") from None
    if not isinstance(data, dict) or not isinstance(data.get("lscpu"), list):
        raise ValueError('no "lscpu" list')

    values: dict[str, object] = {}
    for entry in _flatten_lscpu(data["lscpu"]):
        field = str(entry.get("field", "")).strip().rstrip(":").strip()
        attr = LSCPU_FIELDS.get(field)
        if attr is None or attr in values:  # first occurrence wins
            continue
        raw = entry.get("data")
        if raw is None:
            continue
        if attr in _LSCPU_INT_ATTRS:
            try:
                values[attr] = int(str(raw).strip())
            except ValueError:
                continue
        else:
            values[attr] = str(raw).strip()
    return CpuInfo(**values)


_NUMA_CPUS = re.compile(r"^node\s+(\d+)\s+cpus:\s*(.*)$")
_NUMA_SIZE = re.compile(r"^node\s+(\d+)\s+(size|free):\s*(\d+)\s*MB")
_NUMA_DISTANCE_ROW = re.compile(r"^\s*(\d+):\s*([\d\s]+)$")


def parse_numactl_hardware(text: str) -> Numa:
    """Parse `numactl --hardware`.

    Lines: "available: 2 nodes (0-1)", "node 0 cpus: 0 1 2 ...", "node 0 size: 257579 MB",
    "node 0 free: 200000 MB", then a "node distances:" header row and one row per node.
    Raise ValueError if there are no "node N cpus:" lines.
    """
    nodes: dict[int, NumaNode] = {}
    distances: list[list[int]] = []
    in_distances = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("node distances"):
            in_distances = True
            continue
        if in_distances:
            row = _NUMA_DISTANCE_ROW.match(line)
            if row:
                distances.append([int(x) for x in row.group(2).split()])
            continue
        if m := _NUMA_CPUS.match(stripped):
            node_id = int(m.group(1))
            cpus = [int(c) for c in m.group(2).split() if c.isdigit()]
            nodes[node_id] = NumaNode(id=node_id, cpus=cpus)
        elif (m := _NUMA_SIZE.match(stripped)) and int(m.group(1)) in nodes:
            node = nodes[int(m.group(1))]
            if m.group(2) == "size":
                node.memory_total_mib = int(m.group(3))
            else:
                node.memory_free_mib = int(m.group(3))

    if not nodes:
        raise ValueError('no "node N cpus:" lines')
    return Numa(nodes=[nodes[k] for k in sorted(nodes)], distances=distances)


_IB_CA = re.compile(r"^CA\s+'([^']+)'")
_IB_PORT = re.compile(r"^Port\s+(\d+):")
_IB_FIELDS = {
    "State": "state",
    "Physical state": "physical_state",
    "Rate": "rate_gbps",
    "Link layer": "link_layer",
}


def parse_ibstat(text: str) -> list[InfinibandPort]:
    """Parse `ibstat`: "CA 'mlx5_0'" blocks, each with indented "Port N:" blocks holding
    "State:", "Physical state:", "Rate:" (Gb/s) and "Link layer:"."""
    ports: list[InfinibandPort] = []
    device: str | None = None
    port: InfinibandPort | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if m := _IB_CA.match(line):
            device, port = m.group(1), None
            continue
        if device is None:
            continue
        if m := _IB_PORT.match(stripped):
            port = InfinibandPort(device=device, port=int(m.group(1)))
            ports.append(port)
            continue
        if port is None or ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        attr = _IB_FIELDS.get(key.strip())
        if attr is None:
            continue
        value = value.strip()
        if attr == "rate_gbps":
            try:
                port.rate_gbps = float(value.split()[0])
            except (ValueError, IndexError):
                pass
        else:
            setattr(port, attr, value or None)
    return ports


def has_infiniband() -> bool:
    """True when /sys/class/infiniband exists and lists at least one device (Linux only)."""
    try:
        return os.path.isdir(INFINIBAND_SYSFS) and bool(os.listdir(INFINIBAND_SYSFS))
    except OSError:
        return False


def take_host_snapshot(run: CommandRunner) -> HostSnapshot:
    """Run lscpu and numactl, and ibstat / ibv_devinfo when has_infiniband().

    Nothing here is fatal: a missing command or a parse error leaves that field None. Every
    CommandResult goes into `commands`, so the report can say what was missing and the raw
    text can be re-parsed later.
    """
    snapshot = HostSnapshot()

    lscpu = run(LSCPU_ARGS)
    snapshot.commands.append(lscpu)
    if lscpu.status == "ok":
        try:
            snapshot.cpu = parse_lscpu_json(lscpu.stdout)
        except ValueError:
            pass

    numactl = run(NUMACTL_ARGS)
    snapshot.commands.append(numactl)
    if numactl.status == "ok":
        try:
            snapshot.numa = parse_numactl_hardware(numactl.stdout)
        except ValueError:
            pass

    if has_infiniband():
        ibstat = run(IBSTAT_ARGS)
        snapshot.commands.append(ibstat)
        if ibstat.status == "ok":
            snapshot.infiniband = parse_ibstat(ibstat.stdout) or None
        snapshot.commands.append(run(IBV_DEVINFO_ARGS))

    return snapshot
