"""GPU-1: every command the capture runs must only read (check_read_only)."""

from __future__ import annotations

import pytest

from migration_eval.gpu import samples, snapshot
from migration_eval.gpu.commands import NotReadOnly, check_read_only
from migration_eval.gpu.cpu_samples import COMPUTE_APPS_ARGS, mpstat_args, pidstat_args
from migration_eval.gpu.host import IBSTAT_ARGS, IBV_DEVINFO_ARGS, LSCPU_ARGS, NUMACTL_ARGS
from migration_eval.gpu.serving import PS_ARGS


def test_every_command_the_code_builds_is_allowed():
    """If a module adds or changes a command, this fails until the allowlist agrees."""
    commands = [
        ["nvidia-smi", "-L"],
        ["nvidia-smi", *snapshot.XML_ARGS],
        ["nvidia-smi", *snapshot.TOPO_ARGS],
        ["nvidia-smi", *snapshot.NVLINK_ARGS],
        ["nvidia-smi", *samples.query_args()],
        ["nvidia-smi", *COMPUTE_APPS_ARGS],
        list(LSCPU_ARGS),
        list(NUMACTL_ARGS),
        list(IBSTAT_ARGS),
        list(IBV_DEVINFO_ARGS),
        list(PS_ARGS),
        mpstat_args(5, 120),
        pidstat_args(5, 120),
    ]
    for argv in commands:
        check_read_only(argv)


@pytest.mark.parametrize(
    "argv",
    [
        [r"C:\Windows\System32\nvidia-smi.exe", "-L"],  # Windows path from shutil.which
        ["/usr/bin/nvidia-smi", "-q", "-x"],
        ["NVIDIA-SMI.EXE", "-L"],
        ["nvidia-smi", "-q", "-x", "-i", "0"],
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv", "--id=0,1"],
        ["ibv_devinfo"],
        ["ibstat", "mlx5_0", "1"],
    ],
)
def test_allowed_variants(argv):
    check_read_only(argv)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["rm", "-rf", "/"],
        ["nvidia-smi"],  # bare nvidia-smi isn't on the list
        ["nvidia-smi", "-pl", "200"],  # set power limit
        ["nvidia-smi", "-r"],  # reset GPU
        ["nvidia-smi", "-pm", "1"],  # persistence mode
        ["nvidia-smi", "-q", "-r"],  # allowed prefix, mutating tail
        ["nvidia-smi", "topo", "-p2p", "r"],
        ["numactl", "--cpunodebind=0", "python"],  # runs a program
        ["numactl", "--hardware", "python"],
        ["pidstat", "-u", "-h", "-l", "-e", "rm", "-rf", "/"],  # -e runs a program
        ["mpstat", "-o", "JSON", "-P", "ALL", "5", "120", ";", "rm"],
        ["lscpu", "-J", "--sysroot", "/tmp"],
        ["ps", "-eo", "pid=", "-C", "x; rm"],
        ["ibv_devinfo", "-v", "-d", "mlx5_0"],
    ],
)
def test_refused(argv):
    with pytest.raises(NotReadOnly):
        check_read_only(argv)


def test_message_names_the_command():
    with pytest.raises(NotReadOnly, match="nvidia-smi -pl 200"):
        check_read_only(["nvidia-smi", "-pl", "200"])
