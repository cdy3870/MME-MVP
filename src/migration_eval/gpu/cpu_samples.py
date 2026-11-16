"""CPU use during the capture (GPU-1): `mpstat` for the whole machine, `pidstat` per process.

Both run in the background for the whole capture, with the same interval and count as the
GPU loop (see commands.BackgroundCommand), so the CPU samples line up with the GPU samples:

    mpstat -o JSON -P ALL <interval> <count>    JSON (sysstat >= 11.5). Per-CPU and "all".
    pidstat -u -h -l <interval> <count>         Text, one line per process per interval, with
                                                epoch timestamps (-h) and full command lines (-l).

Both take whole seconds. If --interval isn't a whole number, round it up for these two and
say so; the GPU samples keep the exact interval.

pidstat lists every process. Keep only the ones that matter, so the profile stays small:
GPU compute processes (`nvidia-smi --query-compute-apps=pid,process_name,used_memory`) and
the top PIDSTAT_TOP_N by mean CPU.
"""

from __future__ import annotations

import json
import math
import re
import shlex
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone

from pydantic import BaseModel

from migration_eval.gpu.commands import BackgroundCommand, CommandResult
from migration_eval.gpu.serving import redact_command_line

PIDSTAT_TOP_N = 10
COMPUTE_APPS_ARGS = ("--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits")


class CpuSample(BaseModel):
    """One CPU (or "all") at one interval, in percent of that CPU's time."""

    taken_at: datetime
    cpu: str  # "all", "0", "1", ...
    usr: float
    sys: float
    iowait: float | None = None
    irq: float | None = None
    soft: float | None = None
    steal: float | None = None
    idle: float


class ProcessCpuSample(BaseModel):
    taken_at: datetime
    pid: int
    command: str  # full command line, with secrets redacted (see serving.redact_command_line)
    cpu_pct: float  # can exceed 100 for multi-threaded processes
    cpu: int | None = None  # the core it last ran on; compared against GPU NUMA affinity (GPU-5)
    uses_gpu: bool = False


def whole_seconds(interval_s: float) -> int:
    """mpstat/pidstat interval: interval_s rounded up to a whole second, at least 1."""
    return max(1, math.ceil(interval_s))


def mpstat_args(interval_s: float, count: int) -> list[str]:
    return ["mpstat", "-o", "JSON", "-P", "ALL", str(whole_seconds(interval_s)), str(count)]


def pidstat_args(interval_s: float, count: int) -> list[str]:
    return ["pidstat", "-u", "-h", "-l", str(whole_seconds(interval_s)), str(count)]


# --- mpstat ------------------------------------------------------------------------------------

_MPSTAT_DATE_FORMATS = ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y")
_MPSTAT_TIME_FORMATS = ("%H:%M:%S", "%I:%M:%S %p")


def _parse_date(text: str) -> date:
    for fmt in _MPSTAT_DATE_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised mpstat date {text!r}")


def _parse_time(text: str) -> time:
    for fmt in _MPSTAT_TIME_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt).time()
        except ValueError:
            continue
    raise ValueError(f"unrecognised mpstat time {text!r}")


def _mpstat_entries(text: str) -> tuple[str, list[dict]]:
    """(host date, complete statistics entries), tolerating a cut-off tail."""
    try:
        data = json.loads(text)
        host = data["sysstat"]["hosts"][0]
        return host["date"], list(host.get("statistics") or [])
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        pass

    # Cut off: find the date and the statistics array, then decode entries one at a time.
    date_match = re.search(r'"date"\s*:\s*"([^"]+)"', text)
    stats_match = re.search(r'"statistics"\s*:\s*\[', text)
    if date_match is None or stats_match is None:
        raise ValueError("not mpstat JSON output")
    decoder = json.JSONDecoder()
    entries = []
    pos = stats_match.end()
    while True:
        while pos < len(text) and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(text) or text[pos] == "]":
            break
        try:
            entry, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break  # the cut-off entry
        if isinstance(entry, dict):
            entries.append(entry)
    return date_match.group(1), entries


def parse_mpstat_json(text: str) -> list[CpuSample]:
    """Parse `mpstat -o JSON`: sysstat.hosts[0].statistics[*] with "timestamp" and "cpu-load".

    The timestamp is a local time of day ("10:00:05", or "10:00:05 AM" in some locales) and
    the date is on the host ("date": "11/11/2026"); combine them, and roll the date over
    when the time goes backwards past midnight. A capture stopped early leaves the JSON
    unterminated: parse the complete statistics entries and drop the tail.
    """
    date_text, entries = _mpstat_entries(text)
    day = _parse_date(date_text)
    local_tz = datetime.now().astimezone().tzinfo

    samples: list[CpuSample] = []
    previous: time | None = None
    for entry in entries:
        try:
            clock = _parse_time(str(entry["timestamp"]))
        except (KeyError, ValueError):
            continue
        if previous is not None and clock < previous:
            day += timedelta(days=1)
        previous = clock
        taken_at = datetime.combine(day, clock, tzinfo=local_tz)
        for load in entry.get("cpu-load") or []:
            try:
                samples.append(
                    CpuSample(
                        taken_at=taken_at,
                        cpu=str(load["cpu"]),
                        usr=load["usr"],
                        sys=load["sys"],
                        iowait=load.get("iowait"),
                        irq=load.get("irq"),
                        soft=load.get("soft"),
                        steal=load.get("steal"),
                        idle=load["idle"],
                    )
                )
            except (KeyError, ValueError):
                continue
    return samples


# --- pidstat -----------------------------------------------------------------------------------


def _redact_command(command: str) -> str:
    try:
        argv = shlex.split(command)
    except ValueError:  # unbalanced quotes
        argv = command.split()
    return shlex.join(redact_command_line(argv))


def parse_pidstat(text: str, top_n: int = PIDSTAT_TOP_N, gpu_pids: frozenset[int] = frozenset()) -> list[ProcessCpuSample]:
    """Parse `pidstat -u -h -l`.

    Find the "# Time ... Command" header and read columns by its names (sysstat versions
    add and drop columns, e.g. %wait). Time is seconds since the epoch. Command is the rest
    of the line and may contain spaces. Keep processes in gpu_pids (marked uses_gpu) plus
    the top_n by mean %CPU; drop the rest. Redact secrets in command lines.
    """
    columns: list[str] | None = None
    rows: list[ProcessCpuSample] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            names = stripped.lstrip("#").split()
            if "Time" in names and "Command" in names and "%CPU" in names:
                columns = names
            continue
        if columns is None:
            continue
        cells = stripped.split(None, len(columns) - 1)
        if len(cells) != len(columns):
            continue  # a cut-off line
        row = dict(zip(columns, cells))
        try:
            pid = int(row["PID"])
            taken_at = datetime.fromtimestamp(int(float(row["Time"])), tz=timezone.utc)
            cpu_pct = float(row["%CPU"].replace(",", "."))
            core = int(row["CPU"]) if "CPU" in row else None
        except (KeyError, ValueError):
            continue
        rows.append(
            ProcessCpuSample(
                taken_at=taken_at,
                pid=pid,
                command=_redact_command(row["Command"]),
                cpu_pct=cpu_pct,
                cpu=core,
                uses_gpu=pid in gpu_pids,
            )
        )

    by_pid: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        by_pid[r.pid].append(r.cpu_pct)
    others = sorted(
        (pid for pid in by_pid if pid not in gpu_pids),
        key=lambda pid: sum(by_pid[pid]) / len(by_pid[pid]),
        reverse=True,
    )
    keep = set(gpu_pids) | set(others[: max(0, top_n)])
    return [r for r in rows if r.pid in keep]


def parse_compute_apps(csv_text: str) -> frozenset[int]:
    """PIDs from `nvidia-smi --query-compute-apps=pid,process_name,used_memory` (noheader)."""
    pids = set()
    for line in csv_text.splitlines():
        first = line.split(",", 1)[0].strip()
        if first.isdigit():
            pids.add(int(first))
    return frozenset(pids)


class CpuSampler:
    """Starts mpstat and pidstat for the capture and parses their output when it ends."""

    def __init__(self, interval_s: float, count: int):
        self.mpstat = BackgroundCommand(mpstat_args(interval_s, count))
        self.pidstat = BackgroundCommand(pidstat_args(interval_s, count))

    def start(self) -> None:
        self.mpstat.start()
        self.pidstat.start()

    def stop(self, gpu_pids: frozenset[int] = frozenset()) -> tuple[list[CpuSample], list[ProcessCpuSample], list[CommandResult]]:
        """Stop both, parse what they printed, and return the samples plus both CommandResults.
        A missing command or unparseable output gives an empty list, not an error."""
        mpstat_result = self.mpstat.stop()
        pidstat_result = self.pidstat.stop()

        cpu_samples: list[CpuSample] = []
        if mpstat_result.stdout:
            try:
                cpu_samples = parse_mpstat_json(mpstat_result.stdout)
            except ValueError:
                pass

        process_samples: list[ProcessCpuSample] = []
        if pidstat_result.stdout:
            process_samples = parse_pidstat(pidstat_result.stdout, gpu_pids=gpu_pids)

        # Command lines can carry secrets; the profile keeps the raw output for re-parsing,
        # so store pidstat's with the same redaction as the parsed rows.
        pidstat_result.stdout = "\n".join(_redact_pidstat_line(line) for line in pidstat_result.stdout.splitlines())

        return cpu_samples, process_samples, [mpstat_result, pidstat_result]


def _redact_pidstat_line(line: str) -> str:
    if not line.strip() or line.lstrip().startswith("#"):
        return line
    return " ".join(redact_command_line(line.split(" ")))
