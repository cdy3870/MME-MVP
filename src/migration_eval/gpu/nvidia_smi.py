"""Find and run `nvidia-smi`.

Everything that starts a process lives here, behind the `Runner` type, so the snapshot and
sampling code can be tested with canned output and no GPU.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Sequence

# Takes the arguments after `nvidia-smi` and returns stdout. Raises NvidiaSmiError on failure.
Runner = Callable[[Sequence[str]], str]

DEFAULT_TIMEOUT_S = 30


class NvidiaSmiError(Exception):
    """`nvidia-smi` is missing, failed, or timed out. The message is shown to the partner."""


def find_nvidia_smi() -> str:
    """Return the path to `nvidia-smi`, or raise NvidiaSmiError.

    Use `shutil.which`. The message should say that `gpu capture` must run on the GPU machine
    itself, and point to `gpu import` (GPU-3) for partners who can't install the CLI there.
    """
    raise NotImplementedError


def make_runner(binary: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> Runner:
    """Return a Runner that calls `binary` with the given arguments.

    Use `subprocess.run(..., capture_output=True, text=True, timeout=timeout_s, check=False)`.
    Raise NvidiaSmiError on a non-zero exit (include stderr) or a timeout. A non-zero exit
    usually means the driver isn't loaded ("NVIDIA-SMI has failed because it couldn't
    communicate with the NVIDIA driver").
    """
    raise NotImplementedError


def check_available(run: Runner) -> None:
    """Fail early: run `nvidia-smi -L` and raise NvidiaSmiError if it lists no GPUs."""
    raise NotImplementedError
