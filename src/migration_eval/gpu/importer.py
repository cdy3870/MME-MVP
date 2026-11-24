"""GPU-3: `gpu import <files>` builds gpu_profile.json from command output the partner ran by hand.

For partners who can't install the CLI on the GPU machine. It must produce the same profile
as `gpu capture`, so it accepts the output of the same commands (GPU-1), and the partner
runs them as PARTNER_COMMANDS says (`gpu import --print-commands` prints it).

Files are recognised by their content, not their names, since partners rename files.
Required: the `-q -x` XML and the samples CSV. Everything else is optional and missing
pieces are recorded the same way `gpu capture` records a missing command.

vLLM (GPU-14): `ps -eo pid=,args=` output gives the launch flags (the same parser as capture),
`curl /version` the version, and a curl loop over /metrics the metrics samples, one file per
server, so several metrics files are accepted.

The samples CSV differs from `gpu capture`'s: the partner runs one long `nvidia-smi -l`
with a header row (`--format=csv,nounits`), so the columns are named. Read it by header,
which also handles either throttle field name (clocks_throttle_reasons / clocks_event_reasons).
"""

from __future__ import annotations

import csv
import io
import json
import re
import statistics
from enum import Enum
from pathlib import Path

from migration_eval.capture.file_checks import FileRejected
from migration_eval.gpu.commands import CommandResult
from migration_eval.gpu.cpu_samples import parse_mpstat_json, parse_pidstat
from migration_eval.gpu.host import (
    IBSTAT_ARGS,
    LSCPU_ARGS,
    NUMACTL_ARGS,
    HostSnapshot,
    parse_ibstat,
    parse_lscpu_json,
    parse_numactl_hardware,
)
from migration_eval.gpu.profile import CaptureSettings, GpuProfile
from migration_eval.gpu.samples import (
    FIELD_ATTRS,
    QUERY_FIELDS,
    THROTTLE_FIELD_NEW,
    THROTTLE_FIELD_OLD,
    Sample,
    _convert,
)
from migration_eval.gpu.serving import ServingSettings, find_vllm_processes, parse_vllm_flags
from migration_eval.gpu.snapshot import Snapshot, parse_nvlink_status, parse_query_xml, parse_topo
from migration_eval.gpu.vllm_metrics import (
    SAMPLE_MARKER,
    VllmMetricsSource,
    looks_like_metrics_file,
    parse_metrics_file,
    settings_from_metrics,
)

_FIELDS_NEW = ",".join(THROTTLE_FIELD_NEW if f == THROTTLE_FIELD_OLD else f for f in QUERY_FIELDS)

PARTNER_COMMANDS = f"""\
Run these on the GPU machine while it serves normal traffic, then send us the files.
All of them only read. Skip any command that isn't installed.

  nvidia-smi -q -x > nvidia_smi_q.xml
  nvidia-smi topo -m > nvidia_smi_topo.txt
  nvidia-smi nvlink -s > nvidia_smi_nvlink.txt
  lscpu -J > lscpu.json
  numactl --hardware > numactl.txt
  ibstat > ibstat.txt                       # only if the machine has InfiniBand

Then start these three together and let them run for 10 minutes:

  nvidia-smi --query-gpu={_FIELDS_NEW} --format=csv,nounits -l 5 > nvidia_smi_samples.csv
  mpstat -o JSON -P ALL 5 120 > mpstat.json
  pidstat -u -h -l 5 120 > pidstat.txt

Stop nvidia-smi with Ctrl-C after 10 minutes (mpstat and pidstat stop by themselves).
Older drivers: if nvidia-smi rejects clocks_event_reasons.active, use
clocks_throttle_reasons.active instead.

If the machine serves with vLLM, also send its launch flags, version and metrics. Use the
server's port in place of 8000, and run the metrics loop alongside the three above:

  ps -eo pid=,args= > ps.txt
  curl -s http://localhost:8000/version > vllm_version.json
  for i in $(seq 120); do echo "{SAMPLE_MARKER} $(date -u +%Y-%m-%dT%H:%M:%SZ)"; curl -s http://localhost:8000/metrics; sleep 5; done > vllm_metrics.txt

With several vLLM servers, run the loop once per server, each into its own file.
"""


class FileKind(str, Enum):
    QUERY_XML = "nvidia-smi -q -x"
    TOPO = "nvidia-smi topo -m"
    NVLINK = "nvidia-smi nvlink -s"
    SAMPLES_CSV = "nvidia-smi --query-gpu"
    LSCPU_JSON = "lscpu -J"
    NUMACTL = "numactl --hardware"
    IBSTAT = "ibstat"
    MPSTAT_JSON = "mpstat -o JSON"
    PIDSTAT = "pidstat"
    PS = "ps -eo pid=,args="
    VLLM_VERSION = "curl /version"
    VLLM_METRICS = "curl /metrics loop"


REPEATABLE_KINDS = frozenset({FileKind.VLLM_METRICS})  # one per vLLM server


REQUIRED_KINDS = frozenset({FileKind.QUERY_XML, FileKind.SAMPLES_CSV})

_SEE_COMMANDS = "Run `gpu import --print-commands` for the commands to run."


def detect_kind(text: str) -> FileKind:
    """Recognise a file by its content.

    Signatures: "<nvidia_smi_log" (XML); a line with "CPU Affinity" (topo); "Link N:" lines
    under "GPU N:" lines (nvlink); a first line starting "timestamp" with "utilization.gpu"
    (samples CSV); a JSON object with an "lscpu" key; "available: N nodes" (numactl);
    "CA '" blocks (ibstat); a JSON object with a "sysstat" key (mpstat); a "# Time" header
    with "%CPU" and "Command" (pidstat); "# mme-sample" lines with vllm: series (the /metrics
    loop); a JSON object with only a "version" key (vLLM's /version); "<pid> <command>" on every
    line (ps). Raise ValueError for anything else.
    """
    text = text.lstrip("﻿")
    stripped = text.lstrip()

    if "<nvidia_smi_log" in text:
        return FileKind.QUERY_XML
    if looks_like_metrics_file(text):
        return FileKind.VLLM_METRICS
    if stripped.startswith("{"):
        # mpstat output may be cut off, so look for the key rather than requiring valid JSON.
        head = stripped[:4096]
        if re.search(r'"lscpu"\s*:', head):
            return FileKind.LSCPU_JSON
        if re.search(r'"sysstat"\s*:', head):
            return FileKind.MPSTAT_JSON
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and set(data) == {"version"}:
            return FileKind.VLLM_VERSION

    lines = text.splitlines()
    first = lines[0].strip() if lines else ""
    if first.startswith("timestamp") and "utilization.gpu" in first:
        return FileKind.SAMPLES_CSV
    if any("CPU Affinity" in line for line in lines):
        return FileKind.TOPO
    if any(re.match(r"^GPU \d+:", line) for line in lines) and any(re.match(r"^\s*Link \d+:", line) for line in lines):
        return FileKind.NVLINK
    if any(re.match(r"^available:\s*\d+ nodes", line) for line in lines):
        return FileKind.NUMACTL
    if any(line.startswith("CA '") for line in lines):
        return FileKind.IBSTAT
    if any(line.lstrip().startswith("#") and "Time" in line and "%CPU" in line and "Command" in line for line in lines):
        return FileKind.PIDSTAT
    nonblank = [line for line in lines if line.strip()]
    if nonblank and all(re.match(r"^\s*\d+\s+\S", line) for line in nonblank):
        return FileKind.PS
    raise ValueError("not the output of any command `gpu import` reads")


_UNITS = re.compile(r"\s*\[[^\]]*\]\s*$")

# The CSV header uses nvidia-smi's long field names, not the short ones we query with.
HEADER_ALIASES = {
    "clocks.current.sm": "clocks.sm",
    "clocks.current.memory": "clocks.mem",
}


def parse_samples_with_header(csv_text: str) -> list[Sample]:
    """Parse `--format=csv,nounits` output that has a header row.

    Header names may still carry units in brackets even with nounits ("utilization.gpu [%]"):
    strip them, then map names through samples.FIELD_ATTRS and reuse the value conversion
    from samples.parse_csv. Ignore unknown columns; require timestamp and index. A capture
    stopped with Ctrl-C may end in a partial line: drop it.
    """
    rows = [row for row in csv.reader(io.StringIO(csv_text.lstrip("﻿")), skipinitialspace=True) if any(c.strip() for c in row)]
    if not rows:
        raise ValueError("empty samples file")

    names = [_UNITS.sub("", cell.strip()) for cell in rows[0]]
    names = [HEADER_ALIASES.get(name, name) for name in names]
    attrs = [FIELD_ATTRS.get(name) for name in names]
    if "taken_at" not in attrs or "index" not in attrs:
        raise ValueError("the header has no timestamp and index columns")

    samples: list[Sample] = []
    body = rows[1:]
    for n, row in enumerate(body):
        last = n == len(body) - 1
        if len(row) != len(names):
            if last:
                break  # cut off by Ctrl-C
            raise ValueError(f"row {n + 2} has {len(row)} columns, the header has {len(names)}")
        try:
            values = {attr: _convert(attr, cell.strip()) for attr, cell in zip(attrs, row) if attr is not None}
            samples.append(Sample.model_validate(values))
        except ValueError:
            if last:
                break
            raise
    return samples


def infer_interval_s(samples: list[Sample]) -> float:
    """The median gap between consecutive timestamps of one GPU, for CaptureSettings."""
    by_gpu: dict[int, list[Sample]] = {}
    for s in samples:
        by_gpu.setdefault(s.index, []).append(s)
    for index in sorted(by_gpu):
        times = sorted(s.taken_at for s in by_gpu[index])
        gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
        if gaps:
            return float(statistics.median(gaps))
    return 0.0


def _read_text(path: Path) -> str:
    """Read a partner file. PowerShell's `>` writes UTF-16, so honour a UTF-16 BOM."""
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8-sig", errors="replace")


def import_files(paths: list[Path], hostname: str | None = None) -> GpuProfile:
    """Read, recognise and parse the files into a GpuProfile with settings.source = "import".

    Raise FileRejected (capture.file_checks) when a required kind is missing, when two files
    are the same kind, or when a file isn't recognised; the message names the file and
    points to `gpu import --print-commands`. Reuse the parsers from snapshot.py, host.py and
    cpu_samples.py, so capture and import can't drift apart. started_at / ended_at come from
    the first and last sample; duration_s is their difference. hostname defaults to "unknown"
    (the XML doesn't carry it).
    """
    texts: dict[FileKind, str] = {}
    sources: dict[FileKind, Path] = {}
    metrics_files: list[tuple[Path, str]] = []
    for path in paths:
        text = _read_text(path)
        try:
            kind = detect_kind(text)
        except ValueError:
            raise FileRejected(f"{path} isn't the output of any command `gpu import` reads. {_SEE_COMMANDS}") from None
        if kind in REPEATABLE_KINDS:
            metrics_files.append((path, text))
            continue
        if kind in texts:
            raise FileRejected(f"{sources[kind]} and {path} are both `{kind.value}` output; send only one. {_SEE_COMMANDS}")
        texts[kind], sources[kind] = text, path

    missing = [kind.value for kind in (FileKind.QUERY_XML, FileKind.SAMPLES_CSV) if kind not in texts]
    if missing:
        raise FileRejected(f"Missing required file(s): {', '.join(f'`{m}` output' for m in missing)}. {_SEE_COMMANDS}")

    try:
        driver_version, cuda_version, gpus = parse_query_xml(texts[FileKind.QUERY_XML])
    except ValueError as exc:
        raise FileRejected(f"{sources[FileKind.QUERY_XML]}: could not read the `nvidia-smi -q -x` output: {exc}.") from None

    raw_topo = texts.get(FileKind.TOPO, "")
    try:
        topology = parse_topo(raw_topo) if raw_topo else None
    except ValueError:
        topology = None
    raw_nvlink = texts.get(FileKind.NVLINK, "")
    try:
        nvlink = parse_nvlink_status(raw_nvlink) if raw_nvlink else None
    except ValueError:
        nvlink = None

    snapshot = Snapshot(
        driver_version=driver_version,
        cuda_version=cuda_version,
        gpus=gpus,
        topology=topology,
        nvlink=nvlink,
        raw_xml=texts[FileKind.QUERY_XML],
        raw_topo=raw_topo,
        raw_nvlink=raw_nvlink,
    )

    try:
        samples = parse_samples_with_header(texts[FileKind.SAMPLES_CSV])
    except ValueError as exc:
        raise FileRejected(f"{sources[FileKind.SAMPLES_CSV]}: could not read the samples CSV: {exc}. {_SEE_COMMANDS}") from None
    if not samples:
        raise FileRejected(f"{sources[FileKind.SAMPLES_CSV]} has a header but no samples.")

    host = _host_snapshot(texts)

    cpu_samples = []
    if FileKind.MPSTAT_JSON in texts:
        try:
            cpu_samples = parse_mpstat_json(texts[FileKind.MPSTAT_JSON])
        except ValueError:
            pass
    process_samples = parse_pidstat(texts[FileKind.PIDSTAT]) if FileKind.PIDSTAT in texts else []
    serving, vllm_metrics = _vllm(texts, sources, metrics_files)

    times = sorted(s.taken_at for s in samples)
    started_at, ended_at = times[0], times[-1]
    return GpuProfile(
        hostname=hostname or "unknown",
        settings=CaptureSettings(
            source="import",
            interval_s=infer_interval_s(samples),
            duration_s=(ended_at - started_at).total_seconds(),
            started_at=started_at,
            ended_at=ended_at,
        ),
        snapshot=snapshot,
        samples=samples,
        host=host,
        cpu_samples=cpu_samples,
        process_samples=process_samples,
        serving=serving,
        vllm_metrics=vllm_metrics,
    )


def _vllm(
    texts: dict[FileKind, str], sources: dict[FileKind, Path], metrics_files: list[tuple[Path, str]]
) -> tuple[list[ServingSettings], list[VllmMetricsSource]]:
    """The serving settings from the ps output, and one metrics source per metrics file, as
    `gpu capture` would store them. The /version file names the version of every server (the
    commands collect it from one). Metrics without ps output still give settings
    (`settings_from_metrics`), as `--vllm-url` does in capture."""
    vllm_metrics: list[VllmMetricsSource] = []
    for path, text in metrics_files:
        try:
            vllm_metrics.append(parse_metrics_file(text, path.name))
        except ValueError as exc:
            raise FileRejected(f"{path}: could not read the vLLM metrics: {exc}. {_SEE_COMMANDS}") from None

    version = None
    if FileKind.VLLM_VERSION in texts:
        value = json.loads(texts[FileKind.VLLM_VERSION]).get("version")
        version = value if isinstance(value, str) else None
    for source in vllm_metrics:
        source.version = version

    serving: list[ServingSettings] = []
    if FileKind.PS in texts:
        ps_text = texts[FileKind.PS]
        found = find_vllm_processes(lambda argv: CommandResult(argv=list(argv), status="ok", stdout=ps_text))
        serving = [parse_vllm_flags(argv, pid) for pid, argv in found]
    if not serving:
        serving = [settings_from_metrics(source) for source in vllm_metrics]
    for settings in serving:
        settings.version = settings.version or version
    return serving, vllm_metrics


def _host_snapshot(texts: dict[FileKind, str]) -> HostSnapshot | None:
    """The host part, from whichever of lscpu / numactl / ibstat were sent; None if none were."""
    host_kinds = {FileKind.LSCPU_JSON: LSCPU_ARGS, FileKind.NUMACTL: NUMACTL_ARGS, FileKind.IBSTAT: IBSTAT_ARGS}
    if not any(kind in texts for kind in host_kinds):
        return None

    host = HostSnapshot(
        commands=[CommandResult(argv=list(args), status="ok", stdout=texts[kind]) for kind, args in host_kinds.items() if kind in texts]
    )
    if FileKind.LSCPU_JSON in texts:
        try:
            host.cpu = parse_lscpu_json(texts[FileKind.LSCPU_JSON])
        except ValueError:
            pass
    if FileKind.NUMACTL in texts:
        try:
            host.numa = parse_numactl_hardware(texts[FileKind.NUMACTL])
        except ValueError:
            pass
    if FileKind.IBSTAT in texts:
        host.infiniband = parse_ibstat(texts[FileKind.IBSTAT]) or None
    return host


def missing_pieces(profile: GpuProfile) -> list[str]:
    """The optional inputs a profile lacks, for the import / capture summary."""
    missing = []
    if profile.snapshot.topology is None:
        missing.append("topology (topo -m)")
    if profile.snapshot.nvlink is None:
        missing.append("NVLink status (nvlink -s)")
    if profile.host is None or profile.host.cpu is None:
        missing.append("CPU info (lscpu -J)")
    if profile.host is None or profile.host.numa is None:
        missing.append("NUMA layout (numactl --hardware)")
    if not profile.cpu_samples:
        missing.append("CPU samples (mpstat)")
    if not profile.process_samples:
        missing.append("process samples (pidstat)")
    uses_vllm = any(s.engine == "vllm" for s in profile.serving)
    if uses_vllm and not any(s.status == "ok" for s in profile.vllm_metrics):
        missing.append("vLLM metrics (curl /metrics loop)")
    if profile.vllm_metrics and not any(s.source == "vllm-process" for s in profile.serving):
        missing.append("vLLM launch flags (ps -eo pid=,args=)")
    return missing
