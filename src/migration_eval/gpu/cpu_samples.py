"""CPU use during the capture (GPU-1): `mpstat` for the whole machine, `pidstat` per process.

Both run in the background for the whole capture, with the same interval and count as the
GPU loop (see commands.BackgroundCommand), so the CPU samples line up with the GPU samples:

    mpstat -o JSON -P ALL <interval> <count>    JSON (sysstat >= 11.5). Per-CPU and "all".
    pidstat -u -h -l <interval> <count>         Text, one line per process per interval, all on
                                                one line (-h), with full command lines (-l).

pidstat's Time column depends on the sysstat version: older ones print seconds since the
epoch with -h, newer ones (12.x, e.g. 12.5.2 on Ubuntu 22.04) print local clock time
("04:26:42", or "04:26:42 PM" in a 12-hour locale). The parser takes either; clock times
are dated from the capture's start, rolling past midnight.

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


_CLOCK = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})(?:\s+([AaPp][Mm]))?(?=\s)")


def _normalize_clock(line: str) -> str:
    """Rewrite a leading "HH:MM:SS[ AM|PM]" as "HH:MM:SS" in 24-hour form, so it is one cell."""
    match = _CLOCK.match(line)
    if match is None or match.group(4) is None:
        return line
    hour = int(match.group(1)) % 12 + (12 if match.group(4).lower() == "pm" else 0)
    return f"{hour:02d}:{match.group(2)}:{match.group(3)}" + line[match.end():]


class _ClockDater:
    """Date clock-time rows from the capture start, moving to the next day when time goes back."""

    def __init__(self, started_at: datetime):
        self.local_start = started_at.astimezone()  # pidstat prints the machine's local time
        self.day = self.local_start.date()
        self.last: datetime | None = None

    def __call__(self, cell: str) -> datetime:
        hh, mm, ss = (int(x) for x in cell.split(":"))
        moment = datetime.combine(self.day, time(hh, mm, ss), tzinfo=self.local_start.tzinfo)
        floor = self.last or (self.local_start - timedelta(minutes=5))  # pidstat starts with the capture
        if moment < floor:
            self.day += timedelta(days=1)
            moment += timedelta(days=1)
        self.last = moment
        return moment.astimezone(timezone.utc)


def parse_pidstat(
    text: str,
    top_n: int = PIDSTAT_TOP_N,
    gpu_pids: frozenset[int] = frozenset(),
    started_at: datetime | None = None,
) -> list[ProcessCpuSample]:
    """Parse `pidstat -u -h -l`.

    Find the "# Time ... Command" header and read columns by its names (sysstat versions
    add and drop columns, e.g. %wait). Time is seconds since the epoch or local clock time
    (see the module docstring); clock-time rows need started_at and are skipped without it.
    Command is the rest
    of the line and may contain spaces. Keep processes in gpu_pids (marked uses_gpu) plus
    the top_n by mean %CPU; drop the rest. Redact secrets in command lines.
    """
    columns: list[str] | None = None
    rows: list[ProcessCpuSample] = []
    date_clock = _ClockDater(started_at) if started_at is not None else None

    for line in text.splitlines():
        stripped = _normalize_clock(line.strip())
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
            if ":" in row["Time"]:
                if date_clock is None:
                    continue
                taken_at = date_clock(row["Time"])
            else:
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
        self.started_at = datetime.now(timezone.utc)  # dates pidstat's clock-time rows
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
            process_samples = parse_pidstat(pidstat_result.stdout, gpu_pids=gpu_pids,
                                            started_at=getattr(self, "started_at", None))

        # Command lines can carry secrets; the profile keeps the raw output for re-parsing,
        # so store pidstat's with the same redaction as the parsed rows.
        pidstat_result.stdout = "\n".join(_redact_pidstat_line(line) for line in pidstat_result.stdout.splitlines())

        return cpu_samples, process_samples, [mpstat_result, pidstat_result]


def _redact_pidstat_line(line: str) -> str:
    if not line.strip() or line.lstrip().startswith("#"):
        return line
    return " ".join(redact_command_line(line.split(" ")))
