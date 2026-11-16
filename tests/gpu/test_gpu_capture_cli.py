"""`migration-eval gpu capture` end to end, with nvidia-smi faked out, plus the profile file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from migration_eval import cli
from migration_eval.capture.file_checks import FileRejected
from migration_eval.gpu import samples
from migration_eval.gpu.commands import CommandResult
from migration_eval.gpu.nvidia_smi import NvidiaSmiError
from migration_eval.gpu.profile import read_profile

# tests/ has no __init__.py files, so pytest puts this directory on sys.path (prepend mode).
from test_gpu_samples import TICK
from test_gpu_snapshot import QUERY_XML, TOPO

LIST = "GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-aaaa)\nGPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-bbbb)\n"


def fake_nvidia_smi(args):
    args = list(args)
    if args == ["-L"]:
        return LIST
    if args == ["-q", "-x"]:
        return QUERY_XML
    if args == ["topo", "-m"]:
        return TOPO
    if args[0].startswith("--query-gpu="):
        return TICK
    raise NvidiaSmiError(f"unexpected args {args}")


@pytest.fixture
def no_wait(monkeypatch):
    monkeypatch.setattr(samples.time, "sleep", lambda s: None)


class FakeCpuSampler:
    def __init__(self, interval_s, count):
        pass

    def start(self):
        pass

    def stop(self, gpu_pids=frozenset()):
        return [], [], []


def not_installed(argv):
    """The host commands (lscpu, numactl, ps, ...) as on a machine without them."""
    return CommandResult(argv=list(argv), status="missing")


def install(monkeypatch, runner=fake_nvidia_smi):
    monkeypatch.setattr(cli, "find_nvidia_smi", lambda: "nvidia-smi")
    monkeypatch.setattr(cli, "make_runner", lambda binary: runner)
    monkeypatch.setattr(cli, "run_system_command", not_installed)
    monkeypatch.setattr(cli, "CpuSampler", FakeCpuSampler)


def test_capture_writes_profile(tmp_path: Path, monkeypatch, no_wait):
    install(monkeypatch)
    out = tmp_path / "gpu_profile.json"
    result = CliRunner().invoke(
        cli.app, ["gpu", "capture", "--duration", "3", "--interval", "1", "--out", str(out), "--no-input"]
    )
    assert result.exit_code == 0, result.output
    assert "2 x NVIDIA H100 80GB HBM3" in result.output

    profile = read_profile(out)
    assert profile.settings.source == "capture"
    assert profile.settings.ticks_planned == 3 and not profile.settings.interrupted
    assert len(profile.samples) == 6
    assert profile.snapshot.topology.links["GPU0"]["GPU1"] == "NV18"


def test_capture_without_nvidia_smi_writes_nothing(tmp_path: Path, monkeypatch):
    def missing():
        raise NvidiaSmiError("nvidia-smi was not found on PATH.")

    monkeypatch.setattr(cli, "find_nvidia_smi", missing)
    out = tmp_path / "gpu_profile.json"
    result = CliRunner().invoke(cli.app, ["gpu", "capture", "--out", str(out)])
    assert result.exit_code == 1
    assert "not found" in result.output
    assert not out.exists()


def test_capture_rejects_interval_longer_than_duration(tmp_path: Path):
    result = CliRunner().invoke(
        cli.app, ["gpu", "capture", "--duration", "5", "--interval", "10", "--out", str(tmp_path / "p.json")]
    )
    assert result.exit_code != 0


def test_capture_with_every_sample_failing_writes_nothing(tmp_path: Path, monkeypatch, no_wait):
    def broken_sampling(args):
        if list(args)[0].startswith("--query-gpu=") and broken_sampling.resolved:
            raise NvidiaSmiError("GPU is lost")
        if list(args)[0].startswith("--query-gpu="):
            broken_sampling.resolved = True
        return fake_nvidia_smi(args)

    broken_sampling.resolved = False
    install(monkeypatch, broken_sampling)
    out = tmp_path / "gpu_profile.json"
    result = CliRunner().invoke(
        cli.app, ["gpu", "capture", "--duration", "2", "--interval", "1", "--out", str(out), "--no-input"]
    )
    assert result.exit_code == 1
    assert not out.exists()


def test_read_profile_rejects_other_schema_version(tmp_path: Path):
    path = tmp_path / "gpu_profile.json"
    path.write_text(json.dumps({"schema_version": "gpu-v9"}), encoding="utf-8")
    with pytest.raises(FileRejected, match="gpu-v9"):
        read_profile(path)


def test_read_profile_rejects_invalid_profile(tmp_path: Path):
    path = tmp_path / "gpu_profile.json"
    path.write_text(json.dumps({"schema_version": "gpu-v0", "hostname": "h"}), encoding="utf-8")
    with pytest.raises(FileRejected, match="not a valid GPU profile"):
        read_profile(path)
