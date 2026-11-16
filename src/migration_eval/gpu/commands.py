"""Run the read-only system commands `gpu capture` collects (GPU-1).

GPU-1 requires that every command only reads. `check_read_only` is the gate: it allows a fixed
set of commands with a fixed set of argument shapes, and refuses everything else, so a new
command can't slip into the capture without someone adding it here on purpose. nvidia-smi
goes through the same gate (`nvidia_smi.make_runner` calls it).

Most of these commands exist only on Linux (lscpu, numactl, mpstat, pidstat, ps, ibv_devinfo,
ibstat). A missing command is not an error: the result has status "missing", the profile keeps
it, and the report says what it couldn't see.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from typing import IO, Literal

from pydantic import BaseModel

DEFAULT_TIMEOUT_S = 30

# Command -> the argument prefixes it may start with. An empty tuple means "no arguments".
# Anything not listed is refused. Keep this list short and every entry obviously read-only.
READ_ONLY_COMMANDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "nvidia-smi": (
        ("-q",),
        ("-L",),
        ("topo", "-m"),
        ("nvlink", "-s"),
        ("--query-gpu=",),  # prefix match on the first argument
        ("--query-compute-apps=",),
    ),
    "lscpu": (("-J",),),
    "numactl": (("--hardware",),),  # any other numactl use runs a program under a policy
    "mpstat": (("-o", "JSON", "-P", "ALL"),),
    "pidstat": (("-u", "-h", "-l"),),
    "ps": (("-eo",),),
    "ibv_devinfo": ((), ("-v",)),
    "ibstat": ((),),
}

# Command -> what each argument after the matched prefix must look like (re.fullmatch).
# Prefixes alone aren't enough: `pidstat -u -h -l -e <program>` starts with an allowed prefix
# but `-e` runs a program. Every argument after the prefix must match, or it's refused.
NO_EXTRA_ARGS = r"(?!)"  # matches nothing
TRAILING_ARGS: dict[str, str] = {
    "nvidia-smi": r"-x|--format=[a-z,]+|-i|--id=[\w:.,-]+|\d+",  # output format and GPU selection
    "lscpu": NO_EXTRA_ARGS,
    "numactl": NO_EXTRA_ARGS,
    "mpstat": r"\d+",  # interval, count
    "pidstat": r"\d+",  # interval, count
    "ps": r"[a-z%]+=?(,[a-z%]+=?)*",  # the -eo format list, e.g. "pid=,args="
    "ibv_devinfo": NO_EXTRA_ARGS,
    "ibstat": r"[\w.-]+",  # optional device name and port, e.g. "mlx5_0 1"
}


class NotReadOnly(ValueError):
    """The command isn't on the read-only allowlist. A programming error, not a partner error."""


class CommandResult(BaseModel):
    """One command's outcome, kept in the profile so the raw output can be re-parsed later."""

    argv: list[str]
    status: Literal["ok", "missing", "failed", "timeout"]
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""


# Takes argv, returns a CommandResult. Injected so parsers and snapshots test without a machine.
CommandRunner = Callable[[Sequence[str]], CommandResult]


def check_read_only(argv: Sequence[str]) -> None:
    """Raise NotReadOnly unless argv is on READ_ONLY_COMMANDS.

    Match the command name (basename, without .exe), then require the arguments to start
    with one of its allowed prefixes. A prefix element ending in "=" matches by prefix
    ("--query-gpu=" matches "--query-gpu=index,uuid"); the others must be equal. A command
    whose allowed prefixes include () may also run with no arguments. Every argument after
    the prefix must match the command's TRAILING_ARGS pattern.
    """
    if not argv:
        raise NotReadOnly("empty command")

    # "C:\Windows\System32\nvidia-smi.exe" and "/usr/bin/nvidia-smi" are both "nvidia-smi".
    name = re.split(r"[\\/]", argv[0])[-1].lower()
    if name.endswith(".exe"):
        name = name[: -len(".exe")]
    args = list(argv[1:])

    allowed = READ_ONLY_COMMANDS.get(name)
    if allowed is None:
        raise NotReadOnly(f"`{name}` is not on the read-only command list")
    trailing = re.compile(TRAILING_ARGS.get(name, NO_EXTRA_ARGS))

    for prefix in allowed:
        if len(args) < len(prefix) or not all(_matches(p, a) for p, a in zip(prefix, args)):
            continue
        if all(trailing.fullmatch(arg) for arg in args[len(prefix):]):
            return

    raise NotReadOnly(f"`{' '.join([name, *args])}` is not an allowed read-only use of {name}")


def _matches(expected: str, arg: str) -> bool:
    """A prefix element ending in "=" matches by prefix; anything else must be equal."""
    return arg.startswith(expected) if expected.endswith("=") else arg == expected


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def run_command(argv: Sequence[str], timeout_s: float = DEFAULT_TIMEOUT_S) -> CommandResult:
    """Check argv with check_read_only, run it, and return the result.

    Status: "missing" when the binary isn't on PATH (shutil.which), "timeout" on
    subprocess.TimeoutExpired, "failed" on a non-zero exit, else "ok". Never raises for
    those cases: the caller records the result and carries on.
    """
    check_read_only(argv)
    argv = list(argv)

    if shutil.which(argv[0]) is None:
        return CommandResult(argv=argv, status="missing")

    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, errors="replace", timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(argv=argv, status="timeout", stdout=_text(exc.stdout), stderr=_text(exc.stderr))
    except OSError as exc:  # vanished or not executable between which() and run()
        return CommandResult(argv=argv, status="missing", stderr=str(exc))

    return CommandResult(
        argv=argv,
        status="ok" if result.returncode == 0 else "failed",
        returncode=result.returncode,
        stdout=result.stdout or "",
        stderr=result.stderr or "",
    )


class BackgroundCommand:
    """A command that runs for the whole capture, e.g. `mpstat -o JSON -P ALL 5 120`.

    mpstat and pidstat take an interval and a count and print as they go, so running them
    alongside the GPU loop gives CPU samples on the same clock without blocking it.

    Output goes to temporary files, not pipes: nobody reads until stop(), and a 10-minute
    mpstat on a many-core machine prints megabytes, which would fill a pipe and stall it.
    """

    def __init__(self, argv: Sequence[str]):
        self.argv = list(argv)
        self.proc: subprocess.Popen | None = None
        self._stdout: IO[bytes] | None = None
        self._stderr: IO[bytes] | None = None

    def start(self) -> None:
        """check_read_only, then start the process with its output going to temporary files.
        If the binary is missing, remember that and make stop() return status "missing"."""
        check_read_only(self.argv)
        if shutil.which(self.argv[0]) is None:
            return
        self._stdout = tempfile.TemporaryFile()
        self._stderr = tempfile.TemporaryFile()
        try:
            self.proc = subprocess.Popen(self.argv, stdout=self._stdout, stderr=self._stderr)
        except OSError:
            self._close()
            self.proc = None

    def stop(self, timeout_s: float = 10) -> CommandResult:
        """Wait for the command to finish (it should, given its count); terminate it if it
        hasn't within timeout_s (e.g. the capture was stopped with Ctrl-C). Return what it
        printed. Partial output is still useful: the parsers must tolerate a cut-off tail."""
        if self.proc is None:
            return CommandResult(argv=self.argv, status="missing")

        timed_out = False
        try:
            self.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

        stdout, stderr = self._read(self._stdout), self._read(self._stderr)
        self._close()
        returncode = self.proc.returncode
        if timed_out:
            status = "timeout"
        else:
            status = "ok" if returncode == 0 else "failed"
        return CommandResult(argv=self.argv, status=status, returncode=returncode, stdout=stdout, stderr=stderr)

    @staticmethod
    def _read(f: IO[bytes] | None) -> str:
        if f is None:
            return ""
        f.seek(0)
        return f.read().decode(errors="replace")

    def _close(self) -> None:
        for f in (self._stdout, self._stderr):
            if f is not None:
                f.close()
        self._stdout = self._stderr = None
