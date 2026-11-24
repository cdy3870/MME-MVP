"""GPU-4: reading vLLM settings from the process, asking for other engines."""

from __future__ import annotations

import pytest

from migration_eval.gpu.commands import CommandResult
from migration_eval.gpu.serving import (
    ask_serving_settings,
    find_vllm_processes,
    parse_vllm_flags,
    redact_command_line,
)


API_SERVER = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "meta-llama/Llama-3.1-8B-Instruct",
    "--gpu-memory-utilization", "0.9",
    "--max-num-seqs=256",
    "--enable-prefix-caching",
    "--api-key", "sk-secret",
]
VLLM_SERVE = ["vllm", "serve", "Qwen/Qwen2.5-7B", "-tp", "2", "--dtype", "bfloat16", "--no-enable-chunked-prefill", "--seed", "1"]

PS_OUTPUT = (
    "    1 /sbin/init\n"
    " 3999 /bin/sh -c \"vllm serve Qwen/Qwen2.5-7B -tp 2 > /var/log/vllm.log\"\n"
    " 4242 python3 -m vllm.entrypoints.openai.api_server --model m --api-key sk-secret\n"
    " 4250 python3 -m vllm.entrypoints.openai.api_server --model m --api-key sk-secret\n"
    " 5000 vllm serve Qwen/Qwen2.5-7B -tp 2\n"
    " 6000 grep vllm\n"
)


def ps(argv):
    return CommandResult(argv=list(argv), status="ok", returncode=0, stdout=PS_OUTPUT)


def test_parse_api_server_flags():
    s = parse_vllm_flags(API_SERVER, pid=4242)
    assert (s.engine, s.source, s.model) == ("vllm", "vllm-process", "meta-llama/Llama-3.1-8B-Instruct")
    assert s.gpu_memory_utilization == 0.9 and s.max_num_seqs == 256
    assert s.enable_prefix_caching is True and s.enable_chunked_prefill is None
    assert (s.batch_size, s.gpu_memory_fraction) == (256, 0.9)


def test_parse_vllm_serve_positional_model_aliases_and_no_flags():
    s = parse_vllm_flags(VLLM_SERVE)
    assert s.model == "Qwen/Qwen2.5-7B"
    assert s.tensor_parallel_size == 2 and s.dtype == "bfloat16" and s.precision == "bfloat16"
    assert s.enable_chunked_prefill is False
    assert s.other_flags == {"--seed": "1"}


def test_secrets_never_stored():
    s = parse_vllm_flags(API_SERVER)
    assert "sk-secret" not in (s.command_line or "")
    assert redact_command_line(["x", "--api-key=abc", "--hf-token", "t"]) == ["x", "--api-key=<redacted>", "--hf-token", "<redacted>"]


def test_find_vllm_processes_keeps_parents_and_skips_grep_and_shell_wrappers():
    found = find_vllm_processes(ps)
    assert [pid for pid, _ in found] == [4242, 5000]


def test_find_vllm_processes_without_ps():
    missing = lambda argv: CommandResult(argv=list(argv), status="missing")  # noqa: E731
    assert find_vllm_processes(missing) == []


def test_ask_for_other_engines():
    answers = iter(["sglang", "0.4.1", "128", "bf16", "", "0.85"])
    s = ask_serving_settings(lambda text, **kw: next(answers))
    assert (s.engine, s.source, s.version, s.batch_size, s.precision) == ("sglang", "asked", "0.4.1", 128, "bf16")
    assert s.max_model_len is None and s.gpu_memory_fraction == 0.85


def test_ask_blank_engine_means_none():
    assert ask_serving_settings(lambda text, **kw: "") is None
