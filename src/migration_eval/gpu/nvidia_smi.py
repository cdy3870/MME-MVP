"""Find and run `nvidia-smi`.

Everything that starts a process lives here, behind the `Runner` type, so the snapshot and
sampling code can be tested with canned output and no GPU.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Sequence

from migration_eval.gpu.commands import check_read_only

# Takes the arguments after `nvidia-smi` and returns stdout. Raises NvidiaSmiError on failure.
Runner = Callable[[Sequence[str]], str]

DEFAULT_TIMEOUT_S = 30


class NvidiaSmiError(Exception):
    """`nvidia-smi` is missing, failed, or timed out. The message is shown to the partner."""


def find_nvidia_smi() -> str:
    """Return the path to `nvidia-smi`, or raise NvidiaSmiError.

    `gpu capture` must run on the GPU machine itself; partners who can't install the CLI there
    can run nvidia-smi by hand and use `gpu import` (GPU-3) instead.
    """
    path = shutil.which("nvidia-smi")
    if path is None:
        raise NvidiaSmiError(
            "nvidia-smi was not found on PATH. Run `gpu capture` on the GPU machine itself, "
            "with the NVIDIA driver installed. If you can't install this CLI there, run "
            "nvidia-smi by hand and use `gpu import` instead."
        )
    return path


def make_runner(binary: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> Runner:
    """Return a Runner that calls `binary` with the given arguments.

    Raises NvidiaSmiError on a non-zero exit, a timeout, or a binary that can't be started. A
    non-zero exit usually means the driver isn't loaded ("NVIDIA-SMI has failed because it
    couldn't communicate with the NVIDIA driver").
    """

    def run(args: Sequence[str]) -> str:
        command = [binary, *args]
        check_read_only(command)  # raises NotReadOnly: a programming error, never the partner's
        shown = " ".join(["nvidia-smi", *args])
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout_s, check=False
            )
        except subprocess.TimeoutExpired:
            raise NvidiaSmiError(f"`{shown}` did not finish within {timeout_s:g}s.") from None
        except OSError as exc:  # binary vanished, not executable, ...
            raise NvidiaSmiError(f"Could not run `{shown}`: {exc}") from None

        if result.returncode != 0:
            # nvidia-smi often prints its error to stdout, not stderr.
            detail = (result.stderr or "").strip() or (result.stdout or "").strip()
            message = f"`{shown}` failed with exit code {result.returncode}."
            if detail:
                message += f" {detail}"
            raise NvidiaSmiError(message)
        return result.stdout

    return run


def check_available(run: Runner) -> None:
    """Fail early: run `nvidia-smi -L` and raise NvidiaSmiError if it lists no GPUs.

    `-L` prints one line per GPU: "GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-...)". MIG
    devices appear as indented "MIG ..." lines under their GPU and are not counted.
    """
    output = run(["-L"])
    gpus = [line for line in output.splitlines() if line.startswith("GPU ")]
    if not gpus:
        raise NvidiaSmiError(
            "nvidia-smi ran but listed no GPUs, so there is nothing to capture. "
            "Check that the GPUs are visible to this machine (and to this container, if any)."
        )
