"""The API's request and response shapes: the contract between the server and the screens.

`frontend/src/live/api-types.ts` mirrors these by hand for now. FastAPI publishes them as an
OpenAPI document at /openapi.json, so the TypeScript side can be generated from that instead;
that is an open decision, and the reason these are plain, simple models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RunState = Literal["queued", "running", "cancelling", "done", "failed", "cancelled"]
Tier = Literal["smoke", "full"]


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelt field is an error, not silently ignored


class ProviderStatus(ApiModel):
    name: str  # "openai", "anthropic", "gemini"
    key_variable: str  # the environment variable the server reads, e.g. "ANTHROPIC_API_KEY"
    key_configured: bool  # never the key itself
    # Where the key is kept: the server's environment (local runs), or the partner's Secrets
    # Manager secret (runs on AWS Batch), named in key_secret.
    key_source: Literal["environment", "secret"] = "environment"
    key_secret: str | None = None


class ComponentInfo(ApiModel):
    name: str
    records: int
    replay_grade: int
    incumbent_provider: str | None = None
    incumbent_model: str | None = None


class UploadInfo(ApiModel):
    id: str
    filename: str  # shown to the user only; never used as a path
    size_bytes: int
    record_count: int
    first_started_at: datetime | None = None
    last_started_at: datetime | None = None
    components: list[ComponentInfo]


class RunRequest(ApiModel):
    upload_id: str
    component: str
    tier: Tier = "smoke"
    spending_cap_usd: float = Field(gt=0)
    allow: list[str] = Field(min_length=1)  # providers candidates may be called on; none is refused
    cost_cap_usd: float | None = Field(default=None, gt=0)
    # 1 replays at the logged pace, so a week of logged traffic takes a week. A higher number
    # compresses the gaps between requests; latency is only comparable at 1 (see run.RunConfig).
    replay_speed: float = Field(default=1.0, gt=0)
    # The UI sets this only after the user has seen what will be sent and to which providers.
    # A request without it is refused, exactly as the CLI refuses to start without confirmation.
    confirmed: bool


class RunStatus(ApiModel):
    id: str
    upload_id: str
    component: str
    state: RunState
    stage: str | None = None  # "prepare", "candidates", "baseline", "candidate <provider>/<model>", "report"
    stages_done: list[str] = []
    spent_usd: float | None = None  # read live from the run's spend meter while it is going
    # Replayed requests of the current stage (the baseline or a candidate), while it is going;
    # None in stages that replay nothing. Early stopping can end a candidate below the total.
    calls_done: int | None = None
    calls_total: int | None = None
    spending_cap_usd: float | None = None
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None  # why a run failed, in words the user can act on
    has_report: bool = False
    batch_job_id: str | None = None  # set when the run executes as an AWS Batch job


class RunSummary(ApiModel):
    """One row of the list of past runs."""

    id: str
    component: str
    state: RunState
    started_at: datetime
    verdicts: dict[str, int] | None = None  # e.g. {"safe_to_swap": 1, "do_not_swap": 2}, once done


class GpuReportInfo(ApiModel):
    """One stored GPU report (GPU-5 to GPU-9), built from an uploaded gpu_profile.json."""

    id: str
    filename: str  # shown to the user only; never used as a path
    hostname: str
    created_at: datetime
    gpu_count: int
    gpu_names: list[str]  # distinct models, e.g. ["NVIDIA H100 80GB HBM3"]
    recommendations: int
    has_costs: bool
    upload_id: str | None = None  # the export whose token counts priced cost per million tokens, if one was picked
