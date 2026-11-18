"""`gpu_profile.json`: the snapshots plus the samples, in one file.

Written by `gpu capture` (GPU-1) and `gpu import` (GPU-3), included by `export`, and read by
the GPU report (GPU-5 to GPU-9). GPU-11 (vLLM /metrics samples) will add a field here.

Still to decide: how `export` includes it. Options are a `--gpu-profile` flag on `export`, or
`export` picking up `gpu_profile.json` from the working directory when present. Either way it
goes into the archive as its own member after `records.jsonl`, and the manifest should say
whether it is there.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from migration_eval.capture.file_checks import FileRejected
from migration_eval.gpu.cpu_samples import CpuSample, ProcessCpuSample
from migration_eval.gpu.host import HostSnapshot
from migration_eval.gpu.idle import IdleWarning
from migration_eval.gpu.samples import Sample
from migration_eval.gpu.serving import ServingSettings
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
    ticks_planned: int | None = None
    ticks_failed: int = 0


class GpuProfile(BaseModel):
    schema_version: Literal["gpu-v0"] = GPU_PROFILE_SCHEMA_VERSION
    hostname: str
    settings: CaptureSettings
    snapshot: Snapshot
    samples: list[Sample]
    # GPU-1 additions. None / empty when not collected (missing command, not Linux, import
    # without those files); the report says what it couldn't see.
    host: HostSnapshot | None = None
    cpu_samples: list[CpuSample] = []
    process_samples: list[ProcessCpuSample] = []
    serving: list[ServingSettings] = []  # GPU-4: one per serving process found or asked about
    idle_warning: IdleWarning | None = None  # GPU-2


def write_profile(profile: GpuProfile, path: Path = DEFAULT_PROFILE_PATH) -> None:
    """Write the profile as indented JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_profile(path: Path) -> GpuProfile:
    """Read and validate a profile.

    Raise FileRejected on invalid JSON, an unknown schema version, or a file that doesn't
    validate, matching how `export` rejects records.
    """
    return parse_profile(path.read_text(encoding="utf-8"), str(path))


def parse_profile(text: str, name: str = "The file") -> GpuProfile:
    """`read_profile` for text already in memory (the server's uploads). `name` starts each message."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise FileRejected(f"{name} is not valid JSON.") from None
    if not isinstance(data, dict):
        raise FileRejected(f"{name} is not a JSON object.")

    version = data.get("schema_version")
    if version != GPU_PROFILE_SCHEMA_VERSION:
        raise FileRejected(
            f"{name} has schema_version {version!r}; this CLI reads {GPU_PROFILE_SCHEMA_VERSION!r}. "
            "Re-run `gpu capture` with this version of the CLI."
        )
    try:
        return GpuProfile.model_validate(data)
    except ValidationError as exc:
        where = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors()[:3])
        raise FileRejected(f"{name} is not a valid GPU profile (check: {where}).") from None
