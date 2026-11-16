"""GPU-1: finding and running nvidia-smi, with subprocess faked out."""

from __future__ import annotations

import subprocess

import pytest

from migration_eval.gpu import nvidia_smi
from migration_eval.gpu.nvidia_smi import (
    NvidiaSmiError,
    check_available,
    find_nvidia_smi,
    make_runner,
)

LIST_TWO = (
    "GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-aaaa)\n"
    "GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-bbbb)\n"
)


def fake_run(monkeypatch, *, returncode=0, stdout="", stderr="", raises=None):
    """Replace subprocess.run; return the list the calls are recorded in."""
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr(nvidia_smi.subprocess, "run", run)
    return calls


def test_find_returns_path(monkeypatch):
    monkeypatch.setattr(nvidia_smi.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    assert find_nvidia_smi() == "/usr/bin/nvidia-smi"


def test_find_missing_points_to_gpu_import(monkeypatch):
    monkeypatch.setattr(nvidia_smi.shutil, "which", lambda name: None)
    with pytest.raises(NvidiaSmiError, match="gpu import"):
        find_nvidia_smi()


def test_runner_passes_args_and_returns_stdout(monkeypatch):
    calls = fake_run(monkeypatch, stdout="<xml/>")
    assert make_runner("/bin/nvidia-smi", timeout_s=7)(["-q", "-x"]) == "<xml/>"
    command, kwargs = calls[0]
    assert command == ["/bin/nvidia-smi", "-q", "-x"]
    assert kwargs["timeout"] == 7
    assert kwargs["capture_output"] and kwargs["text"] and kwargs["check"] is False


def test_runner_nonzero_exit_includes_output(monkeypatch):
    driver_down = "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."
    fake_run(monkeypatch, returncode=9, stdout=driver_down)
    with pytest.raises(NvidiaSmiError, match="exit code 9.*couldn't communicate"):
        make_runner("nvidia-smi")(["-L"])


def test_runner_timeout(monkeypatch):
    fake_run(monkeypatch, raises=subprocess.TimeoutExpired("nvidia-smi", 30))
    with pytest.raises(NvidiaSmiError, match="did not finish within 30s"):
        make_runner("nvidia-smi")(["topo", "-m"])


def test_runner_cannot_start(monkeypatch):
    fake_run(monkeypatch, raises=FileNotFoundError("no such file"))
    with pytest.raises(NvidiaSmiError, match="Could not run"):
        make_runner("nvidia-smi")(["-L"])


def test_check_available_accepts_gpus_and_ignores_mig_lines():
    check_available(lambda args: LIST_TWO + "  MIG 1g.10gb Device 0: (UUID: MIG-cccc)\n")


def test_check_available_rejects_empty_list():
    with pytest.raises(NvidiaSmiError, match="listed no GPUs"):
        check_available(lambda args: "No devices were found\n")
