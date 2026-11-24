"""GPU-14: vLLM's own metrics, sampled during `gpu capture` and stored in gpu_profile.json.

For each vLLM server GPU-4 found (`serving.find_vllm_processes`), work out where it listens from
its launch flags (`--host`, `--port`, `--ssl-certfile`; `target_url`), then:

    GET /version   once, before sampling: the vLLM version, which GPU-4 stores in the serving
                   settings (defaults change between versions, prefix caching for one)
    GET /metrics   every `--interval`, on the same schedule as the nvidia-smi readings

Both are read-only GETs on the partner's own machine, so they keep GPU-1's "only read" rule.
`--vllm-url` replaces the URLs worked out from the flags (vLLM in a container with a remapped
port, behind a proxy, or not visible to `ps`), and `--no-vllm-metrics` skips this entirely.

What is kept per sample, summed across the server's label sets (one per model, or per engine
under data parallel) except KV cache use, which is averaged:

    gauges     running and waiting requests, KV cache use
    counters   preemptions, prompt and generated tokens, prefix cache queries and hits
    histograms time to first token and time per output token, as cumulative buckets plus
               their sum and count, so any percentile over the capture can be computed later

Metric names differ between vLLM versions, so each reading has a list of names to try
(`METRIC_NAMES`), newest first, and every other `vllm:` series is also kept, summed by name, in
`other`, so a name this list doesn't know is still in the profile. Left out of `other`: histogram
buckets (too many), `_created` series (when a counter was created, not a reading) and `_info`
series, whose labels are the point: the probe stores them once in `info`. vLLM's
`cache_config_info` holds the settings the server actually runs with, including the ones left
at their defaults (prefix caching, GPU memory fraction, KV cache size in tokens).

A server that can't be reached doesn't stop the capture: its source is stored with status
"unreachable" and the error, and the report lists vLLM metrics as missing (`assessment._missing`).
`summarize` turns a source's samples into what happened over the capture (rates and latencies
from counter differences); the baseline scorecard (GPU-12) and the recommendations (GPU-15)
read that.

`gpu import` reads the same text from a file the partner collected with a curl loop
(`parse_metrics_file`, see importer.PARTNER_COMMANDS).
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import httpx
from pydantic import BaseModel

from migration_eval.gpu.serving import ServingSettings

DEFAULT_VLLM_PORT = 8000
REQUEST_TIMEOUT_S = 2.0  # a slow /metrics must not hold up the tick schedule for long

# Reading -> metric names to try, newest vLLM first. Counters are listed without `_total`; both
# forms are tried. Units: KV cache use is a 0-1 fraction in every version and stored as a percent.
METRIC_NAMES: dict[str, tuple[str, ...]] = {
    "requests_running": ("vllm:num_requests_running",),
    "requests_waiting": ("vllm:num_requests_waiting",),
    "kv_cache_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "preemptions": ("vllm:num_preemptions",),
    "prompt_tokens": ("vllm:prompt_tokens",),
    "generation_tokens": ("vllm:generation_tokens",),
    "prefix_cache_queries": ("vllm:prefix_cache_queries", "vllm:gpu_prefix_cache_queries"),
    "prefix_cache_hits": ("vllm:prefix_cache_hits", "vllm:gpu_prefix_cache_hits"),
    "prefix_cache_hit_rate": ("vllm:gpu_prefix_cache_hit_rate",),  # older versions: a gauge, 0-1
}
HISTOGRAM_NAMES: dict[str, tuple[str, ...]] = {
    "ttft_seconds": ("vllm:time_to_first_token_seconds",),
    "tpot_seconds": ("vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds",
                     "vllm:request_time_per_output_token_seconds"),
}  # fmt: skip
_AVERAGED = {"kv_cache_usage", "prefix_cache_hit_rate"}  # fractions: averaging across label sets, not summing


# --- the model ------------------------------------------------------------------------------


class Histogram(BaseModel):
    """One histogram at one moment, cumulative since the server started (Prometheus semantics)."""

    name: str  # the metric name it was read from, e.g. vllm:time_to_first_token_seconds
    buckets: dict[str, float]  # upper bound ("0.01", ..., "+Inf") -> requests at or under it
    sum: float
    count: float


class VllmMetricsSample(BaseModel):
    """One /metrics reading. None: this vLLM version doesn't expose the reading."""

    taken_at: datetime
    requests_running: float | None = None
    requests_waiting: float | None = None
    kv_cache_usage_pct: float | None = None
    preemptions: float | None = None  # counters: cumulative since the server started
    prompt_tokens: float | None = None
    generation_tokens: float | None = None
    prefix_cache_queries: float | None = None  # tokens looked up in the prefix cache
    prefix_cache_hits: float | None = None  # of those, tokens found
    prefix_cache_hit_rate_pct: float | None = None  # older versions report a rate instead
    ttft_seconds: Histogram | None = None
    tpot_seconds: Histogram | None = None
    other: dict[str, float] = {}  # every other vllm: series, summed by name (no histogram buckets)


class VllmMetricsSource(BaseModel):
    """One vLLM server: where it was read and what came back."""

    url: str  # base URL, e.g. http://127.0.0.1:8000; "file:<name>" for `gpu import`
    pid: int | None = None  # the process GPU-4 found, None for --vllm-url or an import
    url_from: Literal["flags", "option", "import"]
    status: Literal["ok", "unreachable", "skipped"]
    error: str | None = None  # why it is unreachable or skipped, for the report
    version: str | None = None  # from /version
    model: str | None = None  # the model_name label: what the server serves
    info: dict[str, dict[str, str]] = {}  # *_info series -> their labels, e.g. vllm:cache_config_info
    ticks_failed: int = 0  # readings that failed after the first one succeeded
    samples: list[VllmMetricsSample] = []


# --- Prometheus text format -----------------------------------------------------------------

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(\S+)(?:\s+\S+)?$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Series:
    name: str
    labels: dict[str, str]
    value: float


def parse_prometheus(text: str) -> list[Series]:
    """The series in a Prometheus text exposition. Comments, blank lines and lines that don't
    parse are skipped, so one odd line can't lose a whole reading. NaN values are dropped."""
    out: list[Series] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if match is None:
            continue
        name, labels, raw = match.groups()
        try:
            value = float(raw)  # handles +Inf, -Inf, NaN
        except ValueError:
            continue
        if math.isnan(value):
            continue
        out.append(Series(name, {k: v.replace('\\"', '"').replace("\\\\", "\\") for k, v in _LABEL.findall(labels or "")}, value))
    return out


def _total(series: list[Series], name: str) -> tuple[float, int] | None:
    """Sum and number of label sets for `name` (or `name_total`), or None when absent."""
    for candidate in (name, f"{name}_total"):
        values = [s.value for s in series if s.name == candidate]
        if values:
            return sum(values), len(values)
    return None


def _histogram(series: list[Series], name: str) -> Histogram | None:
    buckets: dict[str, float] = {}
    for s in series:
        if s.name == f"{name}_bucket" and "le" in s.labels:
            buckets[s.labels["le"]] = buckets.get(s.labels["le"], 0.0) + s.value
    total_sum = _total(series, f"{name}_sum")
    total_count = _total(series, f"{name}_count")
    if not buckets or total_sum is None or total_count is None:
        return None
    return Histogram(name=name, buckets=buckets, sum=total_sum[0], count=total_count[0])


def _vllm_series(text: str) -> list[Series]:
    series = [s for s in parse_prometheus(text) if s.name.startswith("vllm:")]
    if not series:  # wrong URL: something else answered, or a proxy's error page
        raise ValueError("the response has no vllm: metrics")
    return series


def model_from_text(text: str) -> str | None:
    """The `model_name` label of the first vllm: series that has one."""
    return next((s.labels["model_name"] for s in parse_prometheus(text) if s.name.startswith("vllm:") and "model_name" in s.labels), None)


def info_from_text(text: str) -> dict[str, dict[str, str]]:
    """The labels of each `vllm:*_info` series (the first label set when there are several),
    minus `engine`, which only numbers data-parallel engines."""
    info: dict[str, dict[str, str]] = {}
    for s in parse_prometheus(text):
        if s.name.startswith("vllm:") and s.name.endswith("_info") and s.name not in info:
            info[s.name] = {k: v for k, v in s.labels.items() if k != "engine"}
    return info


def sample_from_text(text: str, taken_at: datetime) -> VllmMetricsSample:
    """One reading from /metrics text. Raise ValueError when the text has no vllm: series."""
    series = _vllm_series(text)

    values: dict[str, object] = {}
    used: set[str] = set()
    for reading, names in METRIC_NAMES.items():
        for name in names:
            found = _total(series, name)
            if found is not None:
                total, count = found
                values[reading] = total / count if reading in _AVERAGED else total
                used.update({name, f"{name}_total"})
                break
    for field in ("kv_cache_usage", "prefix_cache_hit_rate"):
        if field in values:
            values[f"{field}_pct"] = float(values.pop(field)) * 100  # type: ignore[arg-type]

    for reading, names in HISTOGRAM_NAMES.items():
        for name in names:
            histogram = _histogram(series, name)
            if histogram is not None:
                values[reading] = histogram
                used.update({f"{name}_bucket", f"{name}_sum", f"{name}_count"})
                break

    other: dict[str, float] = {}
    for s in series:
        if s.name not in used and not s.name.endswith(("_bucket", "_created", "_info")):
            other[s.name] = other.get(s.name, 0.0) + s.value
    return VllmMetricsSample(taken_at=taken_at, other=other, **values)


# --- where each server listens --------------------------------------------------------------


def target_url(settings: ServingSettings) -> str:
    """The base URL of the vLLM server whose flags are in `settings` (from GPU-4).

    `--host` defaults to every interface; a wildcard host (0.0.0.0, ::, empty) is read on
    loopback. `--port` defaults to 8000. `--ssl-certfile` means https.
    """
    flags = settings.other_flags
    host = flags.get("--host")
    host = host if isinstance(host, str) and host else "127.0.0.1"
    if host in ("0.0.0.0", "*"):
        host = "127.0.0.1"
    elif host in ("::", "[::]"):
        host = "::1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # an IPv6 address in a URL
    port = flags.get("--port")
    port = port if isinstance(port, str) and port.isdigit() else str(DEFAULT_VLLM_PORT)
    scheme = "https" if flags.get("--ssl-certfile") else "http"
    return f"{scheme}://{host}:{port}"


def normalize_url(url: str) -> str:
    """A --vllm-url as typed -> a base URL: add http:// when there is no scheme, and drop a
    trailing /metrics, /v1 or slash, so `localhost:8000/metrics` and `http://localhost:8000` match."""
    url = url.strip()
    if "://" not in url:
        url = f"http://{url}"
    url = url.rstrip("/")
    for suffix in ("/metrics", "/v1"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url.rstrip("/")


# --- reading --------------------------------------------------------------------------------

Fetch = Callable[[str], str]  # GET url -> body; raises on any failure. Tests pass a fake.


def http_fetch(client: httpx.Client) -> Fetch:
    def fetch(url: str) -> str:
        response = client.get(url)
        response.raise_for_status()
        return response.text

    return fetch


def _error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    if isinstance(exc, httpx.HTTPError):
        return f"{type(exc).__name__}"  # ConnectError, ReadTimeout, ...; the message repeats the URL
    return str(exc)


def probe(source: VllmMetricsSource, fetch: Fetch, now: Callable[[], datetime]) -> VllmMetricsSource:
    """Before sampling: read /version (optional) and one /metrics. A /metrics that fails or has
    no vllm: series marks the source unreachable, with the reason; the capture carries on.
    A skipped source (--no-vllm-metrics) is left alone."""
    if source.status == "skipped":
        return source
    try:
        version = json.loads(fetch(f"{source.url}/version")).get("version")
        source.version = version if isinstance(version, str) else None
    except Exception:  # very old versions have no /version; the metrics still count
        source.version = None
    try:
        text = fetch(f"{source.url}/metrics")
        source.samples.append(sample_from_text(text, now()))
        source.info = info_from_text(text)
        source.model = model_from_text(text)
        source.status = "ok"
    except Exception as exc:
        source.status, source.error = "unreachable", _error(exc)
    return source


class VllmMetricsSampler:
    """Reads /metrics from every reachable source every `interval_s` on a background thread,
    alongside the nvidia-smi loop. Tick n is at start + n * interval, as in `samples.collect`."""

    def __init__(
        self,
        sources: list[VllmMetricsSource],
        interval_s: float,
        fetch: Fetch,
        now: Callable[[], datetime] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.sources = sources
        self.interval_s = interval_s
        self._fetch = fetch
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._clock = clock or time.monotonic
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if any(s.status == "ok" for s in self.sources):
            self._thread = threading.Thread(target=self._run, name="vllm-metrics", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        start = self._clock()
        tick = 1  # tick 0 was the probe
        while not self._stop.wait(max(0.0, start + tick * self.interval_s - self._clock())):
            self.read_once()
            tick += 1

    def read_once(self) -> None:
        for source in self.sources:
            if source.status != "ok":
                continue
            try:
                source.samples.append(sample_from_text(self._fetch(f"{source.url}/metrics"), self._now()))
            except Exception:
                source.ticks_failed += 1

    def stop(self) -> list[VllmMetricsSource]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=REQUEST_TIMEOUT_S * max(1, len(self.sources)) + 1)
        return self.sources


def plan_sources(serving: list[ServingSettings], urls: list[str], enabled: bool) -> list[VllmMetricsSource]:
    """The servers to read: one per vLLM process GPU-4 found, or exactly `urls` (--vllm-url) when
    given. With `enabled` False (--no-vllm-metrics) each found server is stored as skipped, so
    the report can say why there are no metrics."""
    if urls:
        unique = dict.fromkeys(normalize_url(u) for u in urls)  # the same server typed two ways is read once
        planned = [VllmMetricsSource(url=u, url_from="option", status="ok") for u in unique]
    else:
        planned = [VllmMetricsSource(url=target_url(s), pid=s.pid, url_from="flags", status="ok") for s in serving if s.source == "vllm-process"]
    if not enabled:
        for source in planned:
            source.status, source.error = "skipped", "skipped with --no-vllm-metrics"
    return planned


def settings_from_metrics(source: VllmMetricsSource) -> ServingSettings:
    """Serving settings for a server read by --vllm-url whose process `ps` can't see (another
    container). The launch flags are unknown, so only what the server reports is filled: the
    version, the model, and from `cache_config_info` the GPU memory fraction and prefix caching,
    which are the values in effect (defaults included), not flags that were passed."""
    config = source.info.get("vllm:cache_config_info", {})
    try:
        fraction = float(config["gpu_memory_utilization"])
    except (KeyError, ValueError):
        fraction = None
    prefix = {"True": True, "False": False}.get(config.get("enable_prefix_caching", ""))
    return ServingSettings(
        engine="vllm", source="vllm-metrics", version=source.version, model=source.model,
        gpu_memory_utilization=fraction, gpu_memory_fraction=fraction, enable_prefix_caching=prefix,
    )  # fmt: skip


def apply_versions(serving: list[ServingSettings], sources: list[VllmMetricsSource]) -> None:
    """Fill each found process's `version` from its server's /version (GPU-4 left it None)."""
    by_pid = {s.pid: s.version for s in sources if s.pid is not None and s.version}
    for settings in serving:
        if settings.version is None and settings.pid in by_pid:
            settings.version = by_pid[settings.pid]
    if len(serving) == 1 and serving[0].version is None:  # one server, read by --vllm-url
        versions = {s.version for s in sources if s.version}
        if len(versions) == 1:
            serving[0].version = versions.pop()


# --- `gpu import` ---------------------------------------------------------------------------

SAMPLE_MARKER = "# mme-sample"  # the partner's curl loop prints "# mme-sample <UTC time>" before each reading


def looks_like_metrics_file(text: str) -> bool:
    return SAMPLE_MARKER in text and re.search(r"^vllm:", text, re.M) is not None


def parse_metrics_file(text: str, name: str) -> VllmMetricsSource:
    """A file of readings, each after a `# mme-sample <time>` line. A reading that doesn't parse
    (curl failed that tick) counts as failed. Raise ValueError when no reading parses."""
    samples: list[VllmMetricsSample] = []
    info: dict[str, dict[str, str]] = {}
    model: str | None = None
    failed = 0
    blocks = re.split(rf"^{re.escape(SAMPLE_MARKER)}[ \t]*", text, flags=re.M)[1:]
    for block in blocks:
        stamp, _, body = block.partition("\n")
        try:
            taken_at = datetime.fromisoformat(stamp.strip().replace("Z", "+00:00"))
            if taken_at.tzinfo is None:
                taken_at = taken_at.replace(tzinfo=timezone.utc)
            samples.append(sample_from_text(body, taken_at))
        except ValueError:
            failed += 1
        else:
            info = info or info_from_text(body)
            model = model or model_from_text(body)
    if not samples:
        raise ValueError("no reading in it could be read")
    return VllmMetricsSource(url=f"file:{name}", url_from="import", status="ok", ticks_failed=failed, samples=samples, info=info, model=model)


# --- over the capture -----------------------------------------------------------------------


class LatencySummary(BaseModel):
    basis: Literal["measured"] = "measured"
    observations: int  # what the histogram counts over the capture: requests for TTFT, gaps between tokens for inter-token latency
    mean_s: float | None
    p50_s: float | None  # from bucket bounds: the bound the percentile falls under
    p95_s: float | None


class VllmMetricsSummary(BaseModel):
    """One server over the capture. Counters are differences between the first and last reading
    (a server restart, seen as a counter going down, makes them None). Always measured (GPU-8)."""

    basis: Literal["measured"] = "measured"
    url: str
    version: str | None
    samples: int
    seconds: float
    requests_running_mean: float | None
    requests_running_max: float | None
    requests_waiting_mean: float | None
    requests_waiting_max: float | None
    kv_cache_usage_pct_mean: float | None
    kv_cache_usage_pct_max: float | None
    preemptions: float | None
    prompt_tokens_per_s: float | None
    generation_tokens_per_s: float | None
    prefix_cache_hit_rate_pct: float | None
    ttft: LatencySummary | None
    tpot: LatencySummary | None


def _gauge(samples: list[VllmMetricsSample], field: str) -> tuple[float | None, float | None]:
    values = [v for v in (getattr(s, field) for s in samples) if v is not None]
    return (sum(values) / len(values), max(values)) if values else (None, None)


def _delta(first: VllmMetricsSample, last: VllmMetricsSample, field: str) -> float | None:
    a, b = getattr(first, field), getattr(last, field)
    if a is None or b is None or b < a:
        return None
    return b - a


def _percentile(buckets: list[tuple[float, float]], total: float, q: float) -> float | None:
    """The smallest bucket bound with at least q of the requests at or under it."""
    for bound, count in buckets:
        if count >= q * total:
            return None if math.isinf(bound) else bound
    return None


def _latency(first: Histogram | None, last: Histogram | None) -> LatencySummary | None:
    if first is None or last is None or last.count < first.count:
        return None
    count = last.count - first.count
    if count <= 0:
        return LatencySummary(observations=0, mean_s=None, p50_s=None, p95_s=None)
    buckets = sorted((float(le), last.buckets.get(le, 0.0) - first.buckets.get(le, 0.0)) for le in last.buckets)
    return LatencySummary(
        observations=int(count),
        mean_s=(last.sum - first.sum) / count,
        p50_s=_percentile(buckets, count, 0.50),
        p95_s=_percentile(buckets, count, 0.95),
    )


def summarize(source: VllmMetricsSource) -> VllmMetricsSummary | None:
    """What one server did over the capture; None without readings."""
    samples = sorted(source.samples, key=lambda s: s.taken_at)
    if not samples:
        return None
    first, last = samples[0], samples[-1]
    seconds = (last.taken_at - first.taken_at).total_seconds()

    def rate(field: str) -> float | None:
        delta = _delta(first, last, field)
        return delta / seconds if delta is not None and seconds > 0 else None

    hit_rate: float | None = None
    queries, hits = _delta(first, last, "prefix_cache_queries"), _delta(first, last, "prefix_cache_hits")
    if queries and hits is not None:
        hit_rate = hits / queries * 100
    elif queries is None:
        hit_rate = _gauge(samples, "prefix_cache_hit_rate_pct")[0]

    running, waiting, kv = _gauge(samples, "requests_running"), _gauge(samples, "requests_waiting"), _gauge(samples, "kv_cache_usage_pct")
    return VllmMetricsSummary(
        url=source.url,
        version=source.version,
        samples=len(samples),
        seconds=seconds,
        requests_running_mean=running[0], requests_running_max=running[1],
        requests_waiting_mean=waiting[0], requests_waiting_max=waiting[1],
        kv_cache_usage_pct_mean=kv[0], kv_cache_usage_pct_max=kv[1],
        preemptions=_delta(first, last, "preemptions"),
        prompt_tokens_per_s=rate("prompt_tokens"),
        generation_tokens_per_s=rate("generation_tokens"),
        prefix_cache_hit_rate_pct=hit_rate,
        ttft=_latency(first.ttft_seconds, last.ttft_seconds),
        tpot=_latency(first.tpot_seconds, last.tpot_seconds),
    )  # fmt: skip
