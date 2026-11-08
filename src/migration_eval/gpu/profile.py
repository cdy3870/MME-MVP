"""`gpu_profile.json`: the snapshot plus the samples, in one file.

Written by `gpu capture` (GPU-1) and `gpu import` (GPU-3), included by `export`, and read by
the GPU report (GPU-5 to GPU-9). GPU-4 (serving engine and settings) will add a field here.

Still to decide: how `export` includes it. Options are a `--gpu-profile` flag on `export`, or
`export` picking up `gpu_profile.json` from the working directory when present. Either way it
goes into the archive as its own member after `records.jsonl`, and the manifest should say
whether it is there.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from migration_eval.gpu.samples import Sample
from migration_eval.gpu.snapshot import Snapshot

GPU_PROFILE_SCHEMA_VERSION = "gpu-v0"
DEFAULT_PROFILE_PATH = Path("gpu_profile.json")


class CaptureSettings(BaseModel):
    source: Literal["capture", "import"]
    interval_s: float
    duration_s: float  # what was asked for
    started_at: datetime
    ended_at: datetime  # earlier than started_at + duration_s if stopped with Ctrl-C
    interrupted: bool = False


class GpuProfile(BaseModel):
    schema_version: Literal["gpu-v0"] = GPU_PROFILE_SCHEMA_VERSION
    hostname: str
    settings: CaptureSettings
    snapshot: Snapshot
    samples: list[Sample]


def write_profile(profile: GpuProfile, path: Path = DEFAULT_PROFILE_PATH) -> None:
    """Write the profile as JSON (`model_dump_json(indent=2)`), creating parent directories."""
    raise NotImplementedError


def read_profile(path: Path) -> GpuProfile:
    """Read and validate a profile. Raise FileRejected (capture.file_checks) on an unknown
    schema version or a file that doesn't validate, matching how `export` rejects records."""
    raise NotImplementedError
