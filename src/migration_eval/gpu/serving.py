"""GPU-4: the serving engine and its settings, stored in gpu_profile.json.

vLLM: read automatically. Find the running vLLM processes (as `ps aux | grep vllm` would,
via the read-only `ps -eo pid=,args=`) and parse their launch flags. A machine often runs one
vLLM server per GPU or per model, so there can be several.

Other engines: ask in the CLI for the engine, its version and the main settings (batch size,
precision, maximum model length, GPU memory fraction). Every question can be skipped.
Reading other engines automatically comes in v3. `--no-input` skips the questions, for
scripted captures.

The configuration recommendations (GPU-6) compare against these settings. A flag that wasn't
passed is stored as None, meaning "the engine's default". Defaults change between vLLM
versions (prefix caching, for one), so the version matters: GPU-11's /version request is the
reliable source; until then it may be None.

Command lines can carry secrets (vLLM's --api-key, tokens in env-style args). Redact them
before anything is stored or printed.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from migration_eval.gpu.commands import CommandRunner

PS_ARGS = ("ps", "-eo", "pid=,args=")
VLLM_MARKERS = ("vllm serve", "vllm.entrypoints.openai.api_server", "vllm.entrypoints.api_server")
SECRET_FLAG_WORDS = ("key", "token", "secret", "password")
REDACTED = "<redacted>"

# vLLM flag -> ServingSettings field. Booleans also accept the --no-<flag> form.
VLLM_VALUE_FLAGS = {
    "--model": "model",
    "--gpu-memory-utilization": "gpu_memory_utilization",
    "--max-num-seqs": "max_num_seqs",
    "--max-num-batched-tokens": "max_num_batched_tokens",
    "--tensor-parallel-size": "tensor_parallel_size",
    "--pipeline-parallel-size": "pipeline_parallel_size",
    "--dtype": "dtype",
    "--max-model-len": "max_model_len",
    "--quantization": "quantization",
    "--kv-cache-dtype": "kv_cache_dtype",
}
VLLM_BOOL_FLAGS = {
    "--enable-prefix-caching": "enable_prefix_caching",
    "--enable-chunked-prefill": "enable_chunked_prefill",
}
# Short aliases vLLM accepts.
VLLM_ALIASES = {"-tp": "--tensor-parallel-size", "-pp": "--pipeline-parallel-size"}

_INT_FIELDS = {"max_num_seqs", "max_num_batched_tokens", "tensor_parallel_size", "pipeline_parallel_size", "max_model_len"}
_FLOAT_FIELDS = {"gpu_memory_utilization"}


class ServingSettings(BaseModel):
    engine: str  # "vllm", or what the partner typed: "sglang", "tgi", "triton", ...
    source: Literal["vllm-process", "asked"]
    version: str | None = None
    pid: int | None = None
    command_line: str | None = None  # redacted
    model: str | None = None

    # The ticket's generic settings (asked for non-vLLM engines; derived for vLLM).
    batch_size: int | None = None  # vLLM: max_num_seqs
    precision: str | None = None  # vLLM: dtype
    max_model_len: int | None = None
    gpu_memory_fraction: float | None = None  # vLLM: gpu_memory_utilization

    # vLLM launch flags. None = not passed, engine default applies.
    gpu_memory_utilization: float | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    dtype: str | None = None
    quantization: str | None = None
    kv_cache_dtype: str | None = None
    enable_prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    other_flags: dict[str, str | bool] = {}  # flags not listed above, redacted


def _is_secret_name(name: str) -> bool:
    name = name.lower()
    return any(word in name for word in SECRET_FLAG_WORDS)


def redact_command_line(argv: list[str]) -> list[str]:
    """Replace the value of any flag whose name contains a SECRET_FLAG_WORDS word.

    Handle both "--api-key VALUE" and "--api-key=VALUE". Return a new list. Env-style
    arguments ("HF_TOKEN=abc") are redacted the same way.
    """
    out: list[str] = []
    redact_next = False
    for arg in argv:
        if redact_next:
            redact_next = False
            if not arg.startswith("-"):
                out.append(REDACTED)
                continue
        if "=" in arg:
            name, _, _value = arg.partition("=")
            if _is_secret_name(name):
                out.append(f"{name}={REDACTED}")
                continue
        elif arg.startswith("-") and _is_secret_name(arg):
            redact_next = True
        out.append(arg)
    return out


def _split(args: str) -> list[str]:
    try:
        return shlex.split(args)
    except ValueError:  # unbalanced quotes
        return args.split()


def find_vllm_processes(run: CommandRunner) -> list[tuple[int, list[str]]]:
    """(pid, argv) for each process whose command line contains a VLLM_MARKERS entry.

    Run PS_ARGS; each line is "<pid> <args>". Split args with shlex. Skip this CLI's own
    process and anything that is just a `grep`. `ps` missing (Windows) -> []. vLLM's API
    server can spawn worker processes with the same command line: keep the parent only
    (lowest pid among identical command lines).
    """
    result = run(PS_ARGS)
    if result.status != "ok":
        return []

    own_pid = os.getpid()
    by_command: dict[str, int] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid, args = int(parts[0]), parts[1].strip()
        if pid == own_pid or not any(marker in args for marker in VLLM_MARKERS):
            continue
        argv = _split(args)
        if not argv or os.path.basename(argv[0]) in ("grep", "egrep", "rg"):
            continue
        if args not in by_command or pid < by_command[args]:
            by_command[args] = pid

    return sorted((pid, _split(args)) for args, pid in by_command.items())


def _convert(field: str, value: str) -> object:
    if field in _INT_FIELDS:
        return int(value)
    if field in _FLOAT_FIELDS:
        return float(value)
    return value


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(value)


def _flag_start(argv: list[str]) -> tuple[int, str | None]:
    """Where the vLLM flags start, and the positional model after `vllm serve` if any."""
    for i, arg in enumerate(argv):
        if os.path.basename(arg) in ("vllm", "vllm.exe") and i + 1 < len(argv) and argv[i + 1] == "serve":
            rest = i + 2
            if rest < len(argv) and not argv[rest].startswith("-"):
                return rest + 1, argv[rest]
            return rest, None
        if "vllm.entrypoints" in arg:
            return i + 1, None
    return 0, None


def parse_vllm_flags(argv: list[str], pid: int | None = None) -> ServingSettings:
    """Turn a vLLM command line into ServingSettings with source "vllm-process".

    Model: the positional argument after `vllm serve`, else --model. Accept "--flag value",
    "--flag=value", VLLM_ALIASES, and "--flag" / "--no-flag" for VLLM_BOOL_FLAGS. Convert
    values to the field's type; keep anything unrecognised in other_flags. Fill the generic
    fields from their vLLM equivalents. Redact before storing command_line / other_flags.
    """
    argv = redact_command_line(list(argv))
    start, positional_model = _flag_start(argv)

    values: dict[str, object] = {}
    other: dict[str, str | bool] = {}
    if positional_model is not None:
        values["model"] = positional_model

    i = start
    while i < len(argv):
        arg = argv[i]
        i += 1
        if not arg.startswith("-"):
            continue  # a stray positional
        name, has_value, inline = arg.partition("=")
        name = VLLM_ALIASES.get(name, name)

        def take_value() -> str | None:
            nonlocal i
            if has_value:
                return inline
            if i < len(argv) and not argv[i].startswith("-"):
                i += 1
                return argv[i - 1]
            return None

        if name in VLLM_BOOL_FLAGS:
            field = VLLM_BOOL_FLAGS[name]
            try:
                values[field] = _parse_bool(inline) if has_value else True
            except ValueError:
                other[arg] = True
        elif name.startswith("--no-") and f"--{name[5:]}" in VLLM_BOOL_FLAGS:
            values[VLLM_BOOL_FLAGS[f"--{name[5:]}"]] = False
        elif name in VLLM_VALUE_FLAGS:
            field = VLLM_VALUE_FLAGS[name]
            value = take_value()
            if value is None:
                other[name] = True
                continue
            if field == "model" and "model" in values:
                continue  # the positional model wins
            try:
                values[field] = _convert(field, value)
            except ValueError:
                other[name] = value
        else:
            value = take_value()
            other[name] = value if value is not None else True

    settings = ServingSettings(
        engine="vllm",
        source="vllm-process",
        pid=pid,
        command_line=shlex.join(argv),
        other_flags=other,
        **values,
    )
    settings.batch_size = settings.max_num_seqs
    settings.precision = settings.dtype
    settings.gpu_memory_fraction = settings.gpu_memory_utilization
    return settings


Prompt = Callable[..., str]  # typer.prompt's shape: prompt(text, default=...) -> str


def _ask(prompt: Prompt, text: str, convert: Callable[[str], object] | None = None) -> object:
    """Ask until the answer is blank (None) or converts; re-ask on a bad value."""
    while True:
        answer = str(prompt(text, default="", show_default=False)).strip()
        if not answer:
            return None
        if convert is None:
            return answer
        try:
            return convert(answer)
        except ValueError:
            text = f"{text.split(' (')[0]} (that didn't parse; leave blank to skip)"


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(value)
    return number


def _fraction(value: str) -> float:
    number = float(value)
    if not 0 < number <= 1:
        raise ValueError(value)
    return number


def ask_serving_settings(prompt: Prompt) -> ServingSettings | None:
    """Ask for the engine and its settings. Blank answers are None.

    Questions: engine (blank -> return None: not self-hosted, or unknown), version, batch
    size, precision, maximum model length, GPU memory fraction (0-1). Re-ask on a value that
    doesn't parse rather than failing the capture.
    """
    engine = _ask(prompt, "Serving engine (e.g. sglang, tgi, triton; blank if none or unknown)")
    if engine is None:
        return None
    return ServingSettings(
        engine=str(engine).lower(),
        source="asked",
        version=_ask(prompt, "Engine version (blank to skip)"),
        batch_size=_ask(prompt, "Maximum batch size (blank to skip)", _positive_int),
        precision=_ask(prompt, "Precision, e.g. bf16, fp8 (blank to skip)"),
        max_model_len=_ask(prompt, "Maximum model length in tokens (blank to skip)", _positive_int),
        gpu_memory_fraction=_ask(prompt, "GPU memory fraction, 0-1 (blank to skip)", _fraction),
    )


def collect_serving_settings(run: CommandRunner, prompt: Prompt, interactive: bool) -> list[ServingSettings]:
    """vLLM processes if any are found; otherwise ask (when interactive); otherwise []."""
    found = find_vllm_processes(run)
    if found:
        return [parse_vllm_flags(argv, pid) for pid, argv in found]
    if interactive:
        asked = ask_serving_settings(prompt)
        return [asked] if asked is not None else []
    return []
