"""GPU-14: vLLM's metrics during `gpu capture` and from `gpu import`, and what the assessment makes of them.

The data/ files are real vLLM 0.33.0 output (Qwen3-8B on an L40), trimmed to keep them small:
vllm_metrics_0_33.txt is one /metrics response without the buckets of histograms we don't read,
and vllm_metrics_loop_0_33.txt is the partner's curl loop (4 readings, 2s apart, under load)
with only the marker lines and vllm: series. OLD_VERSION below is hand-written in the older
names (gpu_cache_usage_perc, gpu_prefix_cache_hit_rate, time_per_output_token_seconds).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from migration_eval import cli
from migration_eval.capture.file_checks import FileRejected
from migration_eval.gpu.assessment import assess
from migration_eval.gpu.commands import CommandResult
from migration_eval.gpu.importer import FileKind, detect_kind, import_files, missing_pieces
from migration_eval.gpu.profile import read_profile
from migration_eval.gpu.serving import ServingSettings, parse_vllm_flags
from migration_eval.gpu.vllm_metrics import (
    VllmMetricsSample,
    VllmMetricsSampler,
    VllmMetricsSource,
    apply_versions,
    normalize_url,
    parse_metrics_file,
    parse_prometheus,
    plan_sources,
    probe,
    sample_from_text,
    settings_from_metrics,
    summarize,
    target_url,
)

# tests/ has no __init__.py files, so pytest puts this directory on sys.path (prepend mode).
from test_gpu_assessment import profile, sample
from test_gpu_capture_cli import FakeCpuSampler, fake_nvidia_smi
from test_gpu_import import SAMPLES_CSV
from test_gpu_snapshot import QUERY_XML

DATA = Path(__file__).parent / "data"
REAL = (DATA / "vllm_metrics_0_33.txt").read_text()
LOOP = (DATA / "vllm_metrics_loop_0_33.txt").read_text()
T0 = datetime(2026, 11, 24, 4, 0, tzinfo=timezone.utc)

OLD_VERSION = """\
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m"} 3.0
vllm:num_requests_waiting{model_name="m"} 1.0
vllm:gpu_cache_usage_perc{model_name="m"} 0.42
vllm:gpu_prefix_cache_hit_rate{model_name="m"} 0.25
vllm:num_preemptions_total{model_name="m"} 2.0
vllm:prompt_tokens_total{model_name="m"} 1000.0
vllm:generation_tokens_total{model_name="m"} 500.0
vllm:time_per_output_token_seconds_bucket{le="0.01",model_name="m"} 10.0
vllm:time_per_output_token_seconds_bucket{le="0.1",model_name="m"} 40.0
vllm:time_per_output_token_seconds_bucket{le="+Inf",model_name="m"} 40.0
vllm:time_per_output_token_seconds_count{model_name="m"} 40.0
vllm:time_per_output_token_seconds_sum{model_name="m"} 1.2
"""

VLLM_ARGV = ["/opt/venv/bin/python", "/opt/venv/bin/vllm", "serve", "Qwen/Qwen3-8B", "--port", "8011", "--max-num-seqs", "32"]
PS_TEXT = " 4319 /bin/bash -c vllm serve Qwen/Qwen3-8B --port 8011\n 4321 " + " ".join(VLLM_ARGV) + "\n"


# --- parsing --------------------------------------------------------------------------------


def test_prometheus_parser_reads_labels_and_skips_comments_and_nan():
    text = '# HELP x y\nvllm:a{k="say \\"hi\\"",le="+Inf"} 2.5\nvllm:b NaN\nnot a line at all\nvllm:c 1e3 1700000000\n'
    series = parse_prometheus(text)
    assert [(s.name, s.value) for s in series] == [("vllm:a", 2.5), ("vllm:c", 1000.0)]
    assert series[0].labels == {"k": 'say "hi"', "le": "+Inf"}


def test_real_0_33_reading():
    s = sample_from_text(REAL, T0)
    assert (s.requests_running, s.requests_waiting, s.kv_cache_usage_pct, s.preemptions) == (0, 0, 0, 0)
    assert (s.prompt_tokens, s.generation_tokens) == (255, 480)
    assert (s.prefix_cache_queries, s.prefix_cache_hits, s.prefix_cache_hit_rate_pct) == (255, 48, None)
    assert s.ttft_seconds.name == "vllm:time_to_first_token_seconds" and s.ttft_seconds.count == 12
    assert s.ttft_seconds.buckets["+Inf"] == 12
    assert s.tpot_seconds.name == "vllm:inter_token_latency_seconds" and s.tpot_seconds.count == 468
    # Unknown series are kept by name; buckets, _created timestamps and _info series are not.
    assert "vllm:e2e_request_latency_seconds_sum" in s.other and "vllm:request_success_total" in s.other
    assert not [n for n in s.other if n.endswith(("_bucket", "_created", "_info"))]
    assert "vllm:num_requests_running" not in s.other and "vllm:prompt_tokens_total" not in s.other


def test_older_metric_names():
    s = sample_from_text(OLD_VERSION, T0)
    assert s.kv_cache_usage_pct == pytest.approx(42) and s.prefix_cache_hit_rate_pct == pytest.approx(25)
    assert (s.preemptions, s.prompt_tokens, s.generation_tokens) == (2, 1000, 500)
    assert s.tpot_seconds.name == "vllm:time_per_output_token_seconds" and s.tpot_seconds.count == 40
    assert s.ttft_seconds is None and s.prefix_cache_queries is None


def test_several_label_sets_sum_counts_and_average_cache_use():
    text = (
        'vllm:num_requests_running{engine="0"} 3\nvllm:num_requests_running{engine="1"} 5\n'
        'vllm:kv_cache_usage_perc{engine="0"} 0.2\nvllm:kv_cache_usage_perc{engine="1"} 0.6\n'
    )
    s = sample_from_text(text, T0)
    assert s.requests_running == 8 and s.kv_cache_usage_pct == pytest.approx(40)


def test_text_without_vllm_series_is_rejected():
    with pytest.raises(ValueError, match="no vllm: metrics"):
        sample_from_text("<html>502 Bad Gateway</html>\nprocess_cpu_seconds_total 3\n", T0)


# --- where each server listens --------------------------------------------------------------


@pytest.mark.parametrize(
    "flags, url",
    [
        ([], "http://127.0.0.1:8000"),
        (["--host", "0.0.0.0", "--port", "9000"], "http://127.0.0.1:9000"),
        (["--host", "::"], "http://[::1]:8000"),
        (["--host=10.0.0.5", "--port=8011"], "http://10.0.0.5:8011"),
        (["--ssl-certfile", "/c.pem", "--ssl-keyfile", "/k.pem"], "https://127.0.0.1:8000"),
    ],
)
def test_target_url_from_launch_flags(flags, url):
    assert target_url(parse_vllm_flags(["vllm", "serve", "m", *flags])) == url


@pytest.mark.parametrize("typed", ["localhost:8000", "http://localhost:8000/", "http://localhost:8000/metrics", "localhost:8000/v1"])
def test_normalize_url(typed):
    assert normalize_url(typed) == "http://localhost:8000"


def test_plan_sources_from_flags_option_or_skipped():
    serving = [parse_vllm_flags(VLLM_ARGV, pid=4321), ServingSettings(engine="sglang", source="asked")]
    (from_flags,) = plan_sources(serving, [], enabled=True)
    assert (from_flags.url, from_flags.pid, from_flags.url_from, from_flags.status) == ("http://127.0.0.1:8011", 4321, "flags", "ok")

    planned = plan_sources(serving, ["localhost:9000", "localhost:9000/metrics", "h:1"], enabled=True)
    assert [(s.url, s.pid, s.url_from) for s in planned] == [("http://localhost:9000", None, "option"), ("http://h:1", None, "option")]

    (skipped,) = plan_sources(serving, [], enabled=False)
    assert skipped.status == "skipped" and "--no-vllm-metrics" in skipped.error


# --- reading --------------------------------------------------------------------------------


def fake_fetch(responses: dict[str, str | Exception]):
    calls: list[str] = []

    def fetch(url: str) -> str:
        calls.append(url)
        answer = responses.get(url, httpx.ConnectError("refused"))
        if isinstance(answer, Exception):
            raise answer
        return answer

    fetch.calls = calls
    return fetch


def test_probe_reads_version_model_info_and_first_sample():
    fetch = fake_fetch({"http://h:1/version": '{"version":"0.33.0"}', "http://h:1/metrics": REAL})
    source = probe(VllmMetricsSource(url="http://h:1", url_from="option", status="ok"), fetch, lambda: T0)
    assert (source.status, source.version, source.model, len(source.samples)) == ("ok", "0.33.0", "Qwen/Qwen3-8B", 1)
    config = source.info["vllm:cache_config_info"]
    assert config["enable_prefix_caching"] == "True" and config["gpu_memory_utilization"] == "0.85" and "engine" not in config


def test_probe_without_version_endpoint_still_reads_metrics():
    source = probe(VllmMetricsSource(url="http://h:1", url_from="flags", status="ok"), fake_fetch({"http://h:1/metrics": OLD_VERSION}), lambda: T0)
    assert source.status == "ok" and source.version is None


@pytest.mark.parametrize(
    "answer, error",
    [
        (httpx.ConnectError("refused"), "ConnectError"),
        (httpx.HTTPStatusError("x", request=httpx.Request("GET", "http://h:1"), response=httpx.Response(404)), "HTTP 404"),
        ("<html>a proxy page</html>", "the response has no vllm: metrics"),
    ],
)
def test_probe_marks_unreachable_with_the_reason(answer, error):
    source = probe(VllmMetricsSource(url="http://h:1", url_from="flags", status="ok"), fake_fetch({"http://h:1/metrics": answer}), lambda: T0)
    assert (source.status, source.error, source.samples) == ("unreachable", error, [])


def test_probe_leaves_skipped_sources_alone():
    fetch = fake_fetch({})
    probe(VllmMetricsSource(url="http://h:1", url_from="flags", status="skipped"), fetch, lambda: T0)
    assert fetch.calls == []


def test_sampler_reads_ok_sources_and_counts_failures():
    good = VllmMetricsSource(url="http://good", url_from="flags", status="ok")
    flaky = VllmMetricsSource(url="http://flaky", url_from="flags", status="ok")
    down = VllmMetricsSource(url="http://down", url_from="flags", status="unreachable")
    fetch = fake_fetch({"http://good/metrics": REAL, "http://flaky/metrics": httpx.ReadTimeout("slow")})
    sampler = VllmMetricsSampler([good, flaky, down], 5, fetch, now=lambda: T0)
    sampler.read_once()
    sampler.read_once()
    assert (len(good.samples), flaky.ticks_failed, len(flaky.samples)) == (2, 2, 0)
    assert "http://down/metrics" not in fetch.calls


def test_sampler_thread_runs_until_stopped():
    source = VllmMetricsSource(url="http://good", url_from="flags", status="ok")
    sampler = VllmMetricsSampler([source], 0.01, fake_fetch({"http://good/metrics": REAL}))
    sampler.start()
    deadline = datetime.now() + timedelta(seconds=5)
    while len(source.samples) < 3 and datetime.now() < deadline:
        pass
    sampler.stop()
    count = len(source.samples)
    assert count >= 3
    assert len(sampler.stop()[0].samples) == count  # stopped: no more readings


def test_sampler_without_reachable_sources_starts_no_thread():
    sampler = VllmMetricsSampler([VllmMetricsSource(url="u", url_from="flags", status="unreachable")], 1, fake_fetch({}))
    sampler.start()
    assert sampler._thread is None


def test_versions_go_to_the_matching_process():
    serving = [parse_vllm_flags(VLLM_ARGV, pid=4321), parse_vllm_flags(["vllm", "serve", "m"], pid=7)]
    apply_versions(serving, [VllmMetricsSource(url="u", pid=4321, url_from="flags", status="ok", version="0.33.0")])
    assert [s.version for s in serving] == ["0.33.0", None]


def test_settings_from_metrics_use_what_the_server_reports():
    source = probe(VllmMetricsSource(url="http://h:1", url_from="option", status="ok"),
                   fake_fetch({"http://h:1/version": '{"version":"0.33.0"}', "http://h:1/metrics": REAL}), lambda: T0)  # fmt: skip
    s = settings_from_metrics(source)
    assert (s.engine, s.source, s.version, s.model) == ("vllm", "vllm-metrics", "0.33.0", "Qwen/Qwen3-8B")
    assert (s.gpu_memory_utilization, s.gpu_memory_fraction, s.enable_prefix_caching) == (0.85, 0.85, True)
    assert s.max_num_seqs is None  # launch flags are unknown without the process


# --- the partner's curl loop and the summary ------------------------------------------------


def test_loop_file_has_one_sample_per_marker():
    source = parse_metrics_file(LOOP, "vllm_metrics.txt")
    assert (source.url, source.url_from, len(source.samples), source.ticks_failed) == ("file:vllm_metrics.txt", "import", 4, 0)
    assert [s.requests_running for s in source.samples] == [0, 14, 15, 14]
    assert source.model == "Qwen/Qwen3-8B" and "vllm:cache_config_info" in source.info


def test_loop_file_counts_failed_ticks_and_rejects_files_with_no_reading():
    text = "# mme-sample 2026-11-24T04:00:00Z\ncurl: (7) Failed to connect\n# mme-sample 2026-11-24T04:00:05Z\n" + OLD_VERSION
    source = parse_metrics_file(text, "m.txt")
    assert (len(source.samples), source.ticks_failed) == (1, 1)
    assert source.samples[0].taken_at == datetime(2026, 11, 24, 4, 0, 5, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        parse_metrics_file("# mme-sample 2026-11-24T04:00:00Z\ncurl: (7) Failed to connect\n", "m.txt")


def test_summary_of_the_real_loop():
    s = summarize(parse_metrics_file(LOOP, "m.txt"))
    assert (s.samples, s.seconds) == (4, 6.0)
    assert (s.requests_running_mean, s.requests_running_max, s.requests_waiting_max) == (10.75, 15, 0)
    assert s.kv_cache_usage_pct_max == pytest.approx(0.478, abs=1e-3) and s.preemptions == 0
    assert s.prompt_tokens_per_s == pytest.approx(230.5) and s.generation_tokens_per_s == pytest.approx(542.67, abs=0.01)
    assert s.prefix_cache_hit_rate_pct == pytest.approx(65.94, abs=0.01)
    assert (s.ttft.observations, s.ttft.p50_s, s.ttft.p95_s) == (58, 0.08, 0.08)
    assert s.ttft.mean_s == pytest.approx(0.0631, abs=1e-4)
    assert (s.tpot.observations, s.tpot.p50_s, s.tpot.p95_s) == (3198, 0.025, 0.05)


def reading(i: int, **values) -> VllmMetricsSample:
    return VllmMetricsSample(taken_at=T0 + timedelta(seconds=5 * i), **values)


def test_summary_after_a_server_restart_has_no_counter_rates():
    source = VllmMetricsSource(url="u", url_from="flags", status="ok",
                               samples=[reading(0, prompt_tokens=900, preemptions=4), reading(1, prompt_tokens=50, preemptions=0)])  # fmt: skip
    s = summarize(source)
    assert s.prompt_tokens_per_s is None and s.preemptions is None


def test_summary_with_older_names_uses_the_hit_rate_gauge():
    source = VllmMetricsSource(url="u", url_from="flags", status="ok",
                               samples=[sample_from_text(OLD_VERSION, T0), sample_from_text(OLD_VERSION, T0 + timedelta(seconds=5))])  # fmt: skip
    s = summarize(source)
    assert s.prefix_cache_hit_rate_pct == pytest.approx(25) and s.tpot.observations == 0 and s.ttft is None


def test_no_samples_no_summary():
    assert summarize(VllmMetricsSource(url="u", url_from="flags", status="unreachable")) is None


# --- the assessment -------------------------------------------------------------------------


def gpu_profile_with(serving, sources):
    p = profile([sample(0, i, 50) for i in range(5)])
    p.serving, p.vllm_metrics = serving, sources
    return p


def test_assessment_summarizes_each_read_server():
    result = assess(gpu_profile_with([parse_vllm_flags(VLLM_ARGV, pid=4321)], [parse_metrics_file(LOOP, "m.txt")]))
    assert [v.url for v in result.vllm] == ["file:m.txt"]
    assert not [m for m in result.missing if "vLLM" in m]
    assert not [f for f in result.findings if f.kind == "missing_data" and "vLLM" in f.message]


@pytest.mark.parametrize(
    "sources, reason",
    [
        ([VllmMetricsSource(url="http://127.0.0.1:8000", url_from="flags", status="unreachable", error="ConnectError")], "http://127.0.0.1:8000: ConnectError"),
        ([VllmMetricsSource(url="http://127.0.0.1:8000", url_from="flags", status="skipped", error="skipped with --no-vllm-metrics")], "skipped with --no-vllm-metrics"),
        ([], "not collected"),
    ],
)
def test_assessment_says_when_vllm_metrics_are_missing(sources, reason):
    result = assess(gpu_profile_with([parse_vllm_flags(VLLM_ARGV, pid=4321)], sources))
    assert any(m.startswith("vLLM metrics (") and reason in m for m in result.missing)
    (finding,) = [f for f in result.findings if f.kind == "missing_data" and "vLLM" in f.message]
    assert reason in finding.message and "--vllm-url" in finding.suggestion
    assert result.vllm == []


def test_no_vllm_no_vllm_notes():
    result = assess(gpu_profile_with([ServingSettings(engine="sglang", source="asked")], []))
    assert not [m for m in result.missing if "vLLM" in m]
