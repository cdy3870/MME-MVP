# DO NOT EDIT. Generated from catalog/catalog.schema.json by scripts/generate_models.py.
# To change these models, edit the schema and rerun the script.

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

type Region = Annotated[
    str,
    Field(
        description='A lowercase region identifier such as us or eu.',
        pattern='^[a-z0-9][a-z0-9-]*$',
    ),
]


class Prices(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    input: Annotated[float, Field(ge=0.0)]
    output: Annotated[float, Field(ge=0.0)]
    cached_input: Annotated[
        float | None,
        Field(
            description="Price for input tokens served from the provider's prompt cache, if it has one.",
            ge=0.0,
        ),
    ] = None


class Offering(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    provider: Annotated[
        str,
        Field(
            description="Lowercase provider key, the same value used in records and in the partner's provider allow-list, e.g. openai, anthropic, gemini.",
            examples=['anthropic'],
            pattern='^[a-z0-9][a-z0-9_-]*$',
        ),
    ]
    model: Annotated[
        str,
        Field(
            description="Model id as the provider's API expects it.",
            examples=['claude-haiku-5-5'],
            min_length=1,
        ),
    ]
    context_window: Annotated[
        int,
        Field(
            description='Maximum tokens the model accepts across input and output.',
            ge=1,
        ),
    ]
    tool_support: Annotated[
        bool,
        Field(
            description='Whether the model supports tool/function calling. Stage 1 drops offerings without it when the traffic uses tools.'
        ),
    ]
    regions: Annotated[
        list[Region],
        Field(
            description='Regions where the provider can process requests for this model, as lowercase identifiers such as us or eu. Stage 1 drops offerings that cannot serve a required region.',
            examples=[['us', 'eu']],
            min_length=1,
        ),
    ]
    prices_usd_per_million_tokens: Prices
    as_of: Annotated[
        str,
        Field(
            description='The date (YYYY-MM-DD) the context window, regions and prices were last checked against source_url. Required. The report shows it, and the weekly review keeps it fresh. A pattern rather than format: date, so the generated model and the schema enforce exactly the same thing; the loader also rejects impossible dates such as 2026-02-30.',
            examples=['2026-10-24'],
            pattern='^\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\\d|3[01])$',
        ),
    ]
    source_url: Annotated[
        str,
        Field(
            description='Link to the provider page the figures came from. Must be https. Required, so a reviewer can verify them.',
            examples=['https://example.com/pricing'],
            pattern='^https://\\S+$',
        ),
    ]


class Catalog(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    offerings: Annotated[list[Offering], Field(min_length=1)]
