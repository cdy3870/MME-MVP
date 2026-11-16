"""GPU-1 additions: read-only gate, nvlink, lscpu, numactl, ibstat, mpstat and pidstat parsing."""

from __future__ import annotations

import json

import pytest

from migration_eval.gpu.cpu_samples import parse_mpstat_json, parse_pidstat
from migration_eval.gpu.host import parse_ibstat, parse_lscpu_json, parse_numactl_hardware
from migration_eval.gpu.snapshot import parse_nvlink_status


NVLINK = (
    "GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-aaaa)\n"
    "\t Link 0: 26.562 GB/s\n"
    "\t Link 1: <inactive>\n"
    "GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-bbbb)\n"
    "\t Link 0: 26.562 GB/s\n"
)

LSCPU = json.dumps(
    {
        "lscpu": [
            {"field": "Architecture:", "data": "x86_64"},
            {"field": "CPU(s):", "data": "16"},
            {"field": "Model name:", "data": "AMD EPYC 7763 64-Core Processor", "children": [
                {"field": "Thread(s) per core:", "data": "2"},
                {"field": "Core(s) per socket:", "data": "4"},
                {"field": "Socket(s):", "data": "2"},
            ]},
            {"field": "NUMA node(s):", "data": "2"},
        ]
    }
)

NUMACTL = """\
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 8 9 10 11
node 0 size: 257579 MB
node 0 free: 200000 MB
node 1 cpus: 4 5 6 7 12 13 14 15
node 1 size: 258043 MB
node 1 free: 210000 MB
node distances:
node   0   1
  0:  10  21
  1:  21  10
"""

IBSTAT = """\
CA 'mlx5_0'
\tCA type: MT4123
\tNumber of ports: 1
\tPort 1:
\t\tState: Active
\t\tPhysical state: LinkUp
\t\tRate: 200
\t\tLink layer: InfiniBand
"""

MPSTAT = json.dumps(
    {
        "sysstat": {
            "hosts": [
                {
                    "nodename": "gpu01",
                    "date": "11/11/2026",
                    "number-of-cpus": 2,
                    "statistics": [
                        {
                            "timestamp": "10:00:05",
                            "cpu-load": [
                                {"cpu": "all", "usr": 12.5, "nice": 0.0, "sys": 3.1, "iowait": 0.2, "irq": 0.0, "soft": 0.4, "steal": 0.0, "guest": 0.0, "gnice": 0.0, "idle": 83.8},
                                {"cpu": "0", "usr": 20.0, "nice": 0.0, "sys": 5.0, "iowait": 0.0, "irq": 0.0, "soft": 0.0, "steal": 0.0, "guest": 0.0, "gnice": 0.0, "idle": 75.0},
                            ],
                        }
                    ],
                }
            ]
        }
    }
)

PIDSTAT = """\
Linux 5.15.0 (gpu01) \t11/11/2026 \t_x86_64_\t(16 CPU)

# Time        UID       PID    %usr %system  %guest   %wait    %CPU   CPU  Command
1762855205   1000      4242   95.00    3.00    0.00    0.00   98.00     3  python3 -m vllm.entrypoints.openai.api_server --model m --api-key sk-secret
1762855205      0         1    0.00    0.00    0.00    0.00    0.00     0  /sbin/init
"""


def test_parse_nvlink_status():
    links = parse_nvlink_status(NVLINK)
    assert [link.active for link in links[0]] == [True, False]
    assert links[1][0].speed_gbps == pytest.approx(26.562)


def test_parse_lscpu_flattens_children():
    cpu = parse_lscpu_json(LSCPU)
    assert (cpu.cpus, cpu.sockets, cpu.cores_per_socket, cpu.threads_per_core) == (16, 2, 4, 2)


def test_parse_numactl_hardware():
    numa = parse_numactl_hardware(NUMACTL)
    assert [n.id for n in numa.nodes] == [0, 1]
    assert numa.nodes[1].cpus == [4, 5, 6, 7, 12, 13, 14, 15]
    assert numa.nodes[0].memory_total_mib == 257579
    assert numa.distances == [[10, 21], [21, 10]]


def test_parse_ibstat():
    (port,) = parse_ibstat(IBSTAT)
    assert (port.device, port.port, port.state, port.rate_gbps) == ("mlx5_0", 1, "Active", 200)


def test_parse_mpstat_json():
    samples = parse_mpstat_json(MPSTAT)
    assert [s.cpu for s in samples] == ["all", "0"]
    assert samples[0].usr == 12.5 and samples[0].idle == 83.8
    assert samples[0].taken_at.date().isoformat() == "2026-11-11"


def test_parse_mpstat_tolerates_cut_off_output():
    # Stopped mid-write: the complete statistics entry before the cut must survive.
    two_entries = MPSTAT.replace('"statistics": [', '"statistics": [' + json.dumps(json.loads(MPSTAT)["sysstat"]["hosts"][0]["statistics"][0]) + ", ", 1)
    samples = parse_mpstat_json(two_entries[:-20])
    assert [s.cpu for s in samples] == ["all", "0"]


def test_parse_pidstat_keeps_gpu_processes_and_redacts():
    samples = parse_pidstat(PIDSTAT, top_n=0, gpu_pids=frozenset({4242}))
    assert [s.pid for s in samples] == [4242]
    assert samples[0].uses_gpu and samples[0].cpu == 3
    assert "sk-secret" not in samples[0].command

