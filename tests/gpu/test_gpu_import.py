"""GPU-3: `gpu import` from command output the partner ran by hand."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from migration_eval import cli
from migration_eval.capture.file_checks import FileRejected
from migration_eval.gpu.importer import FileKind, detect_kind, import_files, parse_samples_with_header

# tests/ has no __init__.py files, so pytest puts this directory on sys.path (prepend mode).
from test_gpu_host import LSCPU, MPSTAT, NUMACTL, NVLINK, PIDSTAT
from test_gpu_snapshot import QUERY_XML, TOPO


SAMPLES_CSV = (
    "timestamp, index, utilization.gpu [%], memory.used [MiB], power.draw [W], clocks_event_reasons.active\n"
    "2026/11/07 12:00:00.000, 0, 87, 61234, 512.30, 0x0000000000000000\n"
    "2026/11/07 12:00:00.000, 1, 12, 1024, [N/A], 0x0000000000000020\n"
    "2026/11/07 12:00:05.000, 0, 90, 61300, 520.00, 0x0000000000000004\n"
    "2026/11/07 12:00:05.000, 1, 15, 1030, [N/A], 0x0000000000000000\n"
    "2026/11/07 12:00:10.0"  # cut off by Ctrl-C
)


def test_print_commands_lists_every_command():
    result = CliRunner().invoke(cli.app, ["gpu", "import", "--print-commands"])
    assert result.exit_code == 0
    for command in ("nvidia-smi -q -x", "topo -m", "nvlink -s", "--query-gpu=", "lscpu -J", "numactl --hardware", "mpstat", "pidstat"):
        assert command in result.output


def test_samples_header_uses_long_clock_names():
    # Real header from driver 582.28: nvidia-smi prints the long names for the clock fields.
    text = (
        "timestamp, index, utilization.gpu [%], clocks.current.sm [MHz], clocks.current.memory [MHz]\n"
        "2026/11/14 20:39:22.007, 0, 7, 1493, 5508\n"
    )
    (s,) = parse_samples_with_header(text)
    assert (s.sm_clock_mhz, s.mem_clock_mhz) == (1493, 5508)


@pytest.mark.parametrize(
    "text, kind",
    [
        (QUERY_XML, FileKind.QUERY_XML),
        (TOPO, FileKind.TOPO),
        (NVLINK, FileKind.NVLINK),
        (SAMPLES_CSV, FileKind.SAMPLES_CSV),
        (LSCPU, FileKind.LSCPU_JSON),
        (NUMACTL, FileKind.NUMACTL),
        (MPSTAT, FileKind.MPSTAT_JSON),
        (PIDSTAT, FileKind.PIDSTAT),
    ],
)
def test_detect_kind_by_content(text, kind):
    assert detect_kind(text) is kind


def test_samples_with_header_strip_units_and_drop_partial_line():
    samples = parse_samples_with_header(SAMPLES_CSV)
    assert len(samples) == 4
    assert samples[0].utilization_gpu_pct == 87
    assert samples[1].power_draw_w is None
    assert samples[1].throttle_reasons == 0x20


def test_import_builds_profile(tmp_path: Path):
    paths = []
    for name, text in {"a.xml": QUERY_XML, "b.txt": TOPO, "c.csv": SAMPLES_CSV, "d.txt": NUMACTL}.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
        paths.append(tmp_path / name)
    profile = import_files(paths, hostname="gpu01")
    assert profile.settings.source == "import"
    assert profile.settings.interval_s == 5
    assert len(profile.snapshot.gpus) == 2 and len(profile.samples) == 4
    assert profile.host.numa is not None and profile.host.cpu is None


def test_import_without_samples_is_rejected(tmp_path: Path):
    path = tmp_path / "q.xml"
    path.write_text(QUERY_XML, encoding="utf-8")
    with pytest.raises(FileRejected, match="print-commands"):
        import_files([path])
