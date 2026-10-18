# DO NOT EDIT. Generated from schema/record.v0.schema.json by scripts/generate_models.py.
# To change these models, edit the schema and rerun the script.

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, RootModel


class RecordSource(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    type: Literal['recorder', 'import_jsonl']
    tool_version: Annotated[str | None, Field(min_length=1)] = None


class Latency(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    total_ms: Annotated[
        float | None, Field(description='Request sent to response complete.', ge=0.0)
    ] = None
    time_to_first_token_ms: Annotated[
        float | None, Field(description='Streamed calls only.', ge=0.0)
    ] = None


class Usage(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    input_tokens: Annotated[int | None, Field(ge=0)] = None
    output_tokens: Annotated[int | None, Field(ge=0)] = None
    cached_input_tokens: Annotated[int | None, Field(ge=0)] = None
    reasoning_tokens: Annotated[int | None, Field(ge=0)] = None


class NamedToolChoice(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    type: Literal['tool']
    name: Annotated[str, Field(min_length=1)]


class ResponseFormat(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    type: Literal['text', 'json_object', 'json_schema']
    schema_: Annotated[dict[str, Any] | None, Field(alias='schema')] = None


class TextPart(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    type: Literal['text']
    text: str


class ImagePart(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    type: Literal['image']
    media_type: str | None = None
    url: str | None = None
    data: Annotated[str | None, Field(description='Base64-encoded image bytes.')] = None


class ToolCallPart(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    type: Literal['tool_call']
    id: Annotated[str | None, Field(min_length=1)] = None
    name: Annotated[str, Field(min_length=1)]
    arguments: dict[str, Any] | None = None
    arguments_raw: str | None = None


class ToolResultPart(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    type: Literal['tool_result']
    tool_call_id: Annotated[str, Field(min_length=1)]
    content: str
    is_error: bool | None = None


class ToolDefinition(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    name: Annotated[str, Field(min_length=1)]
    description: str | None = None
    input_schema: Annotated[
        dict[str, Any] | None,
        Field(description="JSON Schema for the tool's arguments."),
    ] = None


class ToolCall(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    id: Annotated[str | None, Field(min_length=1)] = None
    name: Annotated[str, Field(min_length=1)]
    arguments: Annotated[
        dict[str, Any] | None, Field(description='Parsed arguments.')
    ] = None
    arguments_raw: Annotated[
        str | None,
        Field(
            description='Original argument string when the model emitted JSON that does not parse.'
        ),
    ] = None


class ErrorDetail(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    type: str | None = None
    message: str | None = None
    http_status: int | None = None


class RequestParameters(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    temperature: Annotated[float | None, Field(ge=0.0)] = None
    top_p: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    top_k: Annotated[int | None, Field(ge=1)] = None
    max_output_tokens: Annotated[
        int | None,
        Field(description='Needed to compute worst-case cost per call (NFR-7).', ge=1),
    ] = None
    stop: list[str] | None = None
    seed: int | None = None
    response_format: ResponseFormat | None = None
    extra: Annotated[
        dict[str, Any] | None,
        Field(
            description='Provider-specific parameters with no normalized equivalent.'
        ),
    ] = None


class ContentPart(RootModel[TextPart | ImagePart | ToolCallPart | ToolResultPart]):
    root: TextPart | ImagePart | ToolCallPart | ToolResultPart


class ModelResponse(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    status: Literal['success', 'error']
    id: Annotated[str | None, Field(description="Provider's response id.")] = None
    model: Annotated[
        str | None, Field(description='Model id the provider reports having served.')
    ] = None
    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: (
        Literal['stop', 'length', 'tool_calls', 'content_filter', 'other'] | None
    ) = None
    error: ErrorDetail | None = None


class Message(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    role: Literal['system', 'user', 'assistant', 'tool']
    content: str | list[ContentPart]


class ModelRequest(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    messages: list[Message] | None = None
    parameters: RequestParameters | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: Literal['auto', 'none', 'required'] | NamedToolChoice | None = None
    stream: Annotated[
        bool | None, Field(description='Whether the original call was streamed.')
    ] = None


class CallRecord(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    schema_version: Annotated[
        Literal['v0'],
        Field(
            description='Present on every record. Files mixing versions are rejected (REC-5).'
        ),
    ]
    id: Annotated[
        str,
        Field(description='Unique id of this record within an export.', min_length=1),
    ]
    component: Annotated[
        str,
        Field(
            description="Name of the application component that made the call, set by the recorder's component(name) context or at import.",
            min_length=1,
        ),
    ]
    provider: Annotated[
        str,
        Field(
            description="Lowercase provider key, e.g. openai, anthropic, gemini. Matches the catalog and the partner's provider allow-list.",
            pattern='^[a-z0-9][a-z0-9_-]*$',
        ),
    ]
    model: Annotated[
        str, Field(description='Model id as requested by the caller.', min_length=1)
    ]
    started_at: Annotated[
        AwareDatetime,
        Field(
            description='When the request was sent. Replay paces requests from these timestamps (EVL-3).'
        ),
    ]
    completed_at: Annotated[
        AwareDatetime | None,
        Field(
            description='When the response finished (end of stream for streamed calls).'
        ),
    ] = None
    latency: Latency | None = None
    usage: Usage | None = None
    request: ModelRequest | None = None
    response: ModelResponse | None = None
    metadata: Annotated[
        dict[str, str | float | bool | None] | None,
        Field(
            description='Caller-supplied tags, e.g. a request category. Subject to export redaction.'
        ),
    ] = None
    source: RecordSource | None = None
