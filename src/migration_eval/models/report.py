# DO NOT EDIT. Generated from schema/report.v0.schema.json by scripts/generate_models.py.
# To change these models, edit the schema and rerun the script.

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, RootModel


class CatalogInfo(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    sha256: Annotated[str, Field(pattern='^[0-9a-f]{64}$')]
    as_of: Annotated[
        str,
        Field(
            description="The oldest as_of date among the catalog's offerings.",
            pattern='^\\d{4}-\\d{2}-\\d{2}$',
        ),
    ]


class LatencySummary(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    p50_ms: Annotated[
        float | None, Field(description='Null when no call succeeded.', ge=0.0)
    ]
    p95_ms: Annotated[float | None, Field(ge=0.0)]
    n: Annotated[int, Field(description='Calls the percentiles are over.', ge=0)]
    failed: Annotated[
        int, Field(description='Calls that errored, left out of the percentiles.', ge=0)
    ]


class Incumbent(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    provider: Annotated[str, Field(pattern='^[a-z0-9][a-z0-9_-]*$')]
    model: Annotated[str, Field(min_length=1)]
    latency: LatencySummary
    cost_per_request_usd: Annotated[float | None, Field(ge=0.0)]


class Sample(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    records_available: Annotated[
        int,
        Field(
            description='Replay-grade records for this component in the upload.', ge=0
        ),
    ]
    requests_replayed: Annotated[
        int,
        Field(description='Size of the sample every candidate was drawn from.', ge=0),
    ]
    replay_speed: Annotated[
        float,
        Field(
            description='1 is the logged pace. Latency is only comparable at 1.', gt=0.0
        ),
    ]
    seed: int


class NoiseFloor(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    value: Annotated[
        float | None,
        Field(description='Null when it could not be measured.', ge=0.0, le=1.0),
    ]
    runs: Annotated[int, Field(ge=0)]
    requests: Annotated[int, Field(ge=0)]


class Thresholds(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    min_requests: Annotated[int, Field(ge=0)]
    safe_margin: Annotated[float, Field(ge=0.0)]
    caveat_margin: Annotated[float, Field(ge=0.0)]
    min_category_requests: Annotated[int, Field(ge=0)]


class Agreement(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    agreed: Annotated[int, Field(ge=0)]
    n: Annotated[int, Field(ge=0)]
    rate: Annotated[
        float | None, Field(description='Null when n is 0.', ge=0.0, le=1.0)
    ]
    ci_low: Annotated[float | None, Field(ge=0.0, le=1.0)]
    ci_high: Annotated[float | None, Field(ge=0.0, le=1.0)]


class CategoryAgreement(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    category: Annotated[str, Field(min_length=1)]
    agreement: Agreement


class Cost(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    total_usd: Annotated[float, Field(ge=0.0)]
    per_request_usd: Annotated[
        float | None, Field(description='Null when no call succeeded.', ge=0.0)
    ]


class UnconvertedField(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    field: Annotated[
        str,
        Field(
            description='A path and the reason, e.g. "parameters.seed: Anthropic has no seed".',
            min_length=1,
        ),
    ]
    requests: Annotated[
        int, Field(description='How many replayed requests this affected.', ge=1)
    ]


class Conversion(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    requests: Annotated[int, Field(ge=0)]
    converted_requests: Annotated[int, Field(ge=0)]
    unconverted: list[UnconvertedField]


class CheckFailure(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    check: Literal['error', 'output_schema', 'tool_calls', 'refusal']
    requests: Annotated[int, Field(ge=1)]


class Verdict(
    RootModel[
        Literal['safe_to_swap', 'swap_with_caveats', 'do_not_swap', 'insufficient_data']
    ]
):
    root: Literal[
        'safe_to_swap', 'swap_with_caveats', 'do_not_swap', 'insufficient_data'
    ]


class Candidate(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    provider: Annotated[str, Field(pattern='^[a-z0-9][a-z0-9_-]*$')]
    model: Annotated[str, Field(min_length=1)]
    verdict: Verdict
    explanation: Annotated[
        str,
        Field(
            description='One or two sentences naming the numbers behind the verdict.',
            min_length=1,
        ),
    ]
    stopped_early: Annotated[
        bool,
        Field(
            description='Stopped before the full sample because it was clearly below the noise floor (EVL-2).'
        ),
    ]
    requests_run: Annotated[int, Field(ge=0)]
    agreement: Agreement
    by_category: list[CategoryAgreement]
    latency: LatencySummary
    cost: Cost
    conversion: Conversion
    check_failures: list[CheckFailure]


class Reason(RootModel[str]):
    root: Annotated[str, Field(min_length=1)]


class DroppedCandidate(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    provider: Annotated[str, Field(pattern='^[a-z0-9][a-z0-9_-]*$')]
    model: Annotated[str, Field(min_length=1)]
    stage: Annotated[
        Literal[1, 2],
        Field(description='1 is the capability filter, 2 the cost filter.'),
    ]
    reasons: Annotated[list[Reason], Field(min_length=1)]


class CallFailure(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    where: Annotated[
        str,
        Field(
            description='"baseline" for the current model\'s replays, otherwise the candidate as provider/model.',
            min_length=1,
        ),
    ]
    message: Annotated[str, Field(max_length=240, min_length=1)]
    count: Annotated[
        int, Field(description='How many calls failed with this message.', ge=1)
    ]


class Report(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    schema_version: Literal['v0']
    generated_at: AwareDatetime
    component: Annotated[
        str,
        Field(
            description='The application component this report is about (one report per component).',
            min_length=1,
        ),
    ]
    catalog: CatalogInfo
    incumbent: Incumbent
    sample: Sample
    noise_floor: NoiseFloor
    thresholds: Thresholds
    candidates: Annotated[
        list[Candidate],
        Field(
            description='Every candidate that was replayed, in the order they were run.'
        ),
    ]
    dropped_candidates: Annotated[
        list[DroppedCandidate],
        Field(
            description='Catalog offerings that were never replayed, each with the reasons (CAT-5).'
        ),
    ]
    errors: Annotated[
        list[CallFailure],
        Field(
            description="The most common reasons calls failed, for the baseline and for each candidate, so a run where everything failed says why. Each message is the provider's error text cut to 200 characters; it never includes request headers or keys. An empty list when no call failed."
        ),
    ]
