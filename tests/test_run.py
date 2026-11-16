"""Run an evaluation (RUN-1): the stages, and the whole flow driven by a fake model."""

from __future__ import annotations

import asyncio
import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from migration_eval.catalog import Catalog
from migration_eval.models import CallRecord
from migration_eval.noise_floor import same_output
from migration_eval.profile import Incumbent
from migration_eval.replay import RateLimited, ReplayResult
from migration_eval.run import (
    Baseline,
    RunConfig,
    RunError,
    SpendMeter,
    build_report,
    choose_candidates,
    evaluate_candidate,
    incumbent_offering,
    measure_baseline,
    metered,
    prepare,
    request_outcomes,
    run_evaluation,
    write_report,
)
from migration_eval.metrics import LatencySummary

ROOT = Path(__file__).resolve().parent.parent
REPORT_SCHEMA = json.loads((ROOT / "schema" / "report.v0.schema.json").read_text())
VALIDATOR = Draft202012Validator(REPORT_SCHEMA, format_checker=Draft202012Validator.FORMAT_CHECKER)
T0 = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc)

CATALOG_YAML = """
offerings:
""" + "".join(
    f"""
  - provider: {provider}
    model: {model}
    context_window: 128000
    tool_support: true
    regions: [us]
    prices_usd_per_million_tokens: {{input: {inp}, output: {out}}}
    as_of: "2026-10-24"
    source_url: https://example.com/{model}
"""
    for provider, model, inp, out in [
        ("openai", "gpt-4o", 2.5, 10.0),  # the incumbent
        ("openai", "good-model", 0.15, 0.6),
        ("openai", "bad-model", 0.15, 0.6),
        ("anthropic", "other-model", 3.0, 15.0),
        ("openai", "pricey-model", 500.0, 1500.0),
        ("mistral", "mistral-large", 2.0, 6.0),
    ]
)


def rec(i: int, component="support", **extra) -> CallRecord:
    body = {
        "schema_version": "v0", "id": f"r{i}-{component}", "component": component, "provider": "openai", "model": "gpt-4o",
        "started_at": (T0 + timedelta(seconds=i)).isoformat(),
        "request": {"messages": [{"role": "user", "content": f"q{i}"}]},
        "response": {"status": "success", "text": f"answer to q{i}"},
        "usage": {"input_tokens": 100, "output_tokens": 20},
        "latency": {"total_ms": 500},
    }  # fmt: skip
    body.update(extra)
    return CallRecord.model_validate(body)


def records(n: int) -> list[CallRecord]:
    return [rec(i) for i in range(n)]


@pytest.fixture
def catalog_path(tmp_path: Path) -> Path:
    path = tmp_path / "catalog.yaml"
    path.write_text(CATALOG_YAML)
    return path


def config(**kwargs) -> RunConfig:
    defaults = dict(component="support", tier="full", spending_cap_usd=100.0, allow_list=frozenset({"openai", "anthropic"}), replay_speed=1000.0)
    return RunConfig(**{**defaults, **kwargs})


def factory(calls: list | None = None, answer=None):
    """A fake `make_complete`: every model answers like the incumbent, except `bad-model`."""
    log = calls if calls is not None else []

    def make(source_provider: str):
        async def complete(provider: str, model: str, request) -> ReplayResult:
            log.append((provider, model))
            question = request.messages[-1].content
            text = "something else" if model == "bad-model" else (answer(question) if answer else f"answer to {question}")
            converted = provider != source_provider
            return ReplayResult(text, None, 100, 20, 40.0, None, converted, ("parameters.seed: Anthropic has no seed",) if converted else ())

        return complete

    return make


def run(coro):
    return asyncio.run(coro)


# --- prepare ------------------------------------------------------------------------------


def test_prepare_keeps_only_this_components_replay_grade_records():
    failed = rec(900, response={"status": "error", "error": {"http_status": 500}})
    other = rec(901, component="billing")
    prepared = prepare(records(60) + [failed, other], config(tier="full"))
    assert len(prepared.records) == 60 and {r.component for r in prepared.records} == {"support"}
    assert prepared.incumbent == Incumbent("openai", "gpt-4o")
    assert prepared.profile.component == "support"


def test_the_tier_sets_the_sample_size_and_the_seed_makes_it_repeatable():
    assert len(prepare(records(300), config(tier="smoke")).sample) == 50
    assert len(prepare(records(300), config(tier="full")).sample) == 200
    a, b = prepare(records(300), config(seed=3)).sample, prepare(records(300), config(seed=3)).sample
    assert [r.id for r in a] == [r.id for r in b]
    assert [r.id for r in a] == sorted((r.id for r in a), key=lambda i: int(i.split("-")[0][1:]))  # time order


def test_a_small_component_is_sampled_whole():
    assert len(prepare(records(30), config()).sample) == 30


def test_nothing_to_evaluate_is_a_run_error():
    with pytest.raises(RunError, match="no replay-grade records"):
        prepare(records(10), config(component="nope"))
    with pytest.raises(RunError, match="no replay-grade records"):
        prepare([rec(1, response={"status": "error", "error": {"http_status": 500}})], config())


# --- choose_candidates --------------------------------------------------------------------


def test_stage_one_drops_the_incumbent_and_providers_off_the_allow_list(catalog_path):
    prepared = prepare(records(60), config())
    kept, dropped = choose_candidates(catalog_path, prepared, config())
    assert [(o.provider, o.model) for o in kept] == [
        ("openai", "good-model"), ("openai", "bad-model"), ("anthropic", "other-model"), ("openai", "pricey-model"),
    ]  # fmt: skip
    reasons = {(d["provider"], d["model"]): (d["stage"], d["reasons"]) for d in dropped}
    assert reasons[("openai", "gpt-4o")] == (1, ["it is the model the traffic already uses"])
    assert reasons[("mistral", "mistral-large")][0] == 1 and "allow-list" in reasons[("mistral", "mistral-large")][1][0]


def test_stage_two_runs_only_with_a_cost_cap(catalog_path):
    cfg = config(cost_cap_usd=5.0)
    kept, dropped = choose_candidates(catalog_path, prepare(records(60), cfg), cfg)
    assert "pricey-model" not in [o.model for o in kept]
    stage_two = [d for d in dropped if d["stage"] == 2]
    assert [d["model"] for d in stage_two] == ["pricey-model"] and "cap is $5.00" in stage_two[0]["reasons"][0]


def test_an_empty_allow_list_is_a_run_error(catalog_path):
    cfg = config(allow_list=frozenset())
    with pytest.raises(RunError, match="allow-list is empty"):
        choose_candidates(catalog_path, prepare(records(60), cfg), cfg)


def test_a_bad_catalog_is_a_run_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("offerings: [not an offering]")
    with pytest.raises(RunError):
        choose_candidates(bad, prepare(records(60), config()), config())
    with pytest.raises(RunError):
        incumbent_offering(bad, Incumbent("openai", "gpt-4o"))


def test_the_incumbent_is_found_in_the_catalog_for_its_prices(catalog_path):
    assert incumbent_offering(catalog_path, Incumbent("openai", "gpt-4o")).prices_usd_per_million_tokens.input == 2.5
    assert incumbent_offering(catalog_path, Incumbent("openai", "not-listed")) is None


# --- the spend meter ----------------------------------------------------------------------


def priced(input_price=2.5, output_price=10.0):
    return Catalog.model_validate({"offerings": [{
        "provider": "openai", "model": "m", "context_window": 1000, "tool_support": True, "regions": ["us"],
        "prices_usd_per_million_tokens": {"input": input_price, "output": output_price},
        "as_of": "2026-10-24", "source_url": "https://example.com/m"}]}).offerings[0]  # fmt: skip


def test_the_meter_allows_spending_up_to_the_cap_and_no_further():
    meter = SpendMeter(1.0)
    assert meter.can_afford(1.0) and not meter.can_afford(1.01)
    meter.record(0.6)
    assert meter.can_afford(0.4) and not meter.can_afford(0.41) and meter.spent_usd == 0.6


def request_with(limit=None):
    from migration_eval.models.record import ModelRequest

    body = {"messages": [{"role": "user", "content": "hi"}]}
    if limit:
        body["parameters"] = {"max_output_tokens": limit}
    return ModelRequest.model_validate(body)


def ok(text="ok", input_tokens=100, output_tokens=20):
    async def complete(provider, model, request):
        return ReplayResult(text, None, input_tokens, output_tokens, 1.0, None)

    return complete


def test_a_metered_call_records_its_actual_cost():
    meter = SpendMeter(10.0)
    result = run(metered(ok(), meter, priced())("openai", "m", request_with(500)))
    assert result.error is None
    assert meter.spent_usd == pytest.approx((100 * 2.5 + 20 * 10.0) / 1e6) and meter.refused == 0


def test_a_call_that_could_pass_the_cap_is_not_sent():
    calls = []

    async def complete(provider, model, request):
        calls.append(1)
        return ReplayResult("ok", None, 1, 1, 1.0, None)

    meter = SpendMeter(0.001)  # a 4,096-token default answer at $10/M is about $0.04
    result = run(metered(complete, meter, priced())("openai", "m", request_with()))
    assert result.error == "spending cap reached" and calls == [] and meter.refused == 1 and meter.spent_usd == 0


def test_the_worst_case_uses_the_requests_own_output_limit():
    meter = SpendMeter(0.001)
    assert run(metered(ok(), meter, priced())("openai", "m", request_with(10))).error is None  # 10 tokens is cheap
    assert meter.refused == 0


def test_concurrent_calls_reserve_their_worst_case_so_they_cannot_overshoot_together():
    async def slow(provider, model, request):
        await asyncio.sleep(0.01)
        return ReplayResult("ok", None, 1, 1, 1.0, None)

    meter = SpendMeter(0.045)  # room for one 4,096-token worst case ($0.041), not two

    async def both():
        wrapped = metered(slow, meter, priced())
        return await asyncio.gather(wrapped("openai", "m", request_with()), wrapped("openai", "m", request_with()))

    results = run(both())
    assert sorted(r.error is None for r in results) == [False, True] and meter.refused == 1


def test_a_call_that_raises_gives_its_reservation_back():
    async def limited(provider, model, request):
        raise RateLimited(1.0)

    meter = SpendMeter(10.0)
    with pytest.raises(RateLimited):
        run(metered(limited, meter, priced())("openai", "m", request_with()))
    assert meter.spent_usd == pytest.approx(0.0)


# --- measure_baseline ---------------------------------------------------------------------


def test_the_baseline_replays_the_incumbent_three_times_and_measures_it(catalog_path):
    cfg, calls = config(), []
    prepared = prepare(records(40), cfg)
    baseline = run(measure_baseline(prepared, incumbent_offering(catalog_path, prepared.incumbent), factory(calls), cfg))
    assert len(baseline.runs) == 3 and all(len(r) == 40 for r in baseline.runs)
    assert set(calls) == {("openai", "gpt-4o")} and len(calls) == 120
    assert baseline.noise_floor == 1.0
    assert baseline.latency.n == 120 and baseline.latency.p50_ms == 40.0
    assert baseline.cost_per_request_usd == pytest.approx((100 * 2.5 + 20 * 10.0) / 1e6)


def test_an_incumbent_missing_from_the_catalog_has_no_cost():
    cfg = config()
    baseline = run(measure_baseline(prepare(records(20), cfg), None, factory(), cfg))
    assert baseline.cost_per_request_usd is None and baseline.latency.n == 60


def test_a_flaky_incumbent_has_a_noise_floor_below_one():
    cfg, counter = config(), {"n": 0}

    def flaky(question):
        counter["n"] += 1
        return f"v{counter['n'] % 3}"

    baseline = run(measure_baseline(prepare(records(20), cfg), None, factory(answer=flaky), cfg))
    assert baseline.noise_floor < 1


# --- request_outcomes ---------------------------------------------------------------------


def baseline_of(*runs: list[ReplayResult]) -> Baseline:
    return Baseline(list(runs), 1.0, LatencySummary(None, None, 0, 0), None)


def result(text="a", error=None) -> ReplayResult:
    return ReplayResult(text, None, 1, 1, 1.0, error)


def test_a_request_agrees_when_most_incumbent_runs_agree():
    sample = records(3)
    runs = [[result("x"), result("x"), result("x")], [result("x"), result("y"), result("x")], [result("x"), result("y"), result("y")]]
    outcomes, _ = request_outcomes(sample, [result("x")] * 3, baseline_of(*runs))
    assert outcomes == [True, False, True]  # 3 of 3, 1 of 3 and 2 of 3 incumbent runs agree


def test_only_the_requests_the_candidate_ran_are_scored():
    runs = [[result("x")] * 5] * 3
    outcomes, checks = request_outcomes(records(5), [result("x"), result("x")], baseline_of(*runs))
    assert outcomes == [True, True] and len(checks) == 2


def test_a_failed_check_decides_the_request_without_the_comparator():
    schema_record = rec(0, request={
        "messages": [{"role": "user", "content": "q0"}],
        "parameters": {"response_format": {"type": "json_schema", "schema": {"type": "object"}}},
    })  # fmt: skip
    runs = [[result("not json")]] * 3  # the incumbent and candidate say the same words...
    outcomes, checks = request_outcomes([schema_record], [result("not json")], baseline_of(*runs))
    assert outcomes == [False] and checks[0][0].check == "output_schema"  # ...but the candidate broke the schema


def test_a_candidate_that_errored_disagrees():
    outcomes, checks = request_outcomes(records(1), [result(None, error="HTTP 500")], baseline_of(*[[result("x")]] * 3))
    assert outcomes == [False] and checks[0][0].check == "error"


def test_a_custom_comparator_decides_when_the_checks_pass():
    outcomes, _ = request_outcomes(records(1), [result("x")], baseline_of(*[[result("y")]] * 3), comparator=lambda a, b: True)
    assert outcomes == [True]


# --- evaluate_candidate -------------------------------------------------------------------


def evaluate(model, *, cfg=None, n=120, should_cancel=lambda: False, meter=None, provider="openai"):
    cfg = cfg or config()
    prepared = prepare(records(n), cfg)
    baseline = run(measure_baseline(prepared, None, factory(), cfg))
    offering = priced()
    offering = type(offering).model_validate({**offering.model_dump(), "provider": provider, "model": model})
    return run(evaluate_candidate(offering, prepared, baseline, factory(), cfg, same_output, should_cancel, meter)), prepared


def test_a_candidate_that_matches_runs_every_request():
    evaluation, prepared = evaluate("good-model")
    assert evaluation.run.requests_run == 120 and evaluation.run.agreement == 1.0
    assert evaluation.outcomes == [True] * 120 and not evaluation.run.stopped_early and not evaluation.cancelled


def test_a_clearly_worse_candidate_is_stopped_early():
    evaluation, _ = evaluate("bad-model")
    assert evaluation.run.stopped_early and evaluation.run.requests_run == 50
    assert evaluation.outcomes == [False] * 50 and not evaluation.cancelled


def test_a_stop_request_ends_the_candidate_and_marks_it_cancelled():
    sent = []
    evaluation, _ = evaluate("good-model", should_cancel=lambda: True)
    assert evaluation.cancelled and all(r.error == "run stopped" for r in evaluation.run.results) and sent == []


def test_a_spent_budget_cancels_the_candidate():
    meter = SpendMeter(cap_usd=1.0, spent_usd=1.0)
    evaluation, _ = evaluate("good-model", meter=meter)
    assert evaluation.cancelled and meter.refused > 0


def test_the_candidate_is_sent_the_logged_providers_name_to_the_factory():
    seen = []

    def spy(source_provider):
        seen.append(source_provider)
        return factory()(source_provider)

    cfg = config()
    prepared = prepare(records(30), cfg)
    baseline = run(measure_baseline(prepared, None, spy, cfg))
    run(evaluate_candidate(priced(), prepared, baseline, spy, cfg))
    assert set(seen) == {"openai"}  # where the traffic was logged from, not the candidate's provider


# --- the whole run ------------------------------------------------------------------------


def full_run(catalog_path, **kwargs):
    stages: list[str] = []
    cfg = kwargs.pop("cfg", config(cost_cap_usd=5.0))
    report = run(run_evaluation(records(kwargs.pop("n", 120)), catalog_path, cfg, kwargs.pop("make", factory()), on_stage=stages.append, now=lambda: NOW, **kwargs))
    return report, stages


def by_model(report):
    return {c.model: c for c in report.candidates}


def test_a_run_produces_a_report_that_satisfies_the_schema(catalog_path):
    report, _ = full_run(catalog_path)
    data = report.model_dump(mode="json", by_alias=True)
    assert not list(VALIDATOR.iter_errors(data)), [e.message for e in VALIDATOR.iter_errors(data)]


def test_the_report_says_what_was_measured(catalog_path):
    report, _ = full_run(catalog_path)
    assert report.component == "support" and report.generated_at == NOW
    assert (report.incumbent.provider, report.incumbent.model) == ("openai", "gpt-4o")
    assert report.noise_floor.value == 1.0 and report.noise_floor.runs == 3 and report.noise_floor.requests == 120
    assert (report.sample.records_available, report.sample.requests_replayed, report.sample.seed) == (120, 120, 1)
    assert report.incumbent.latency.n == 360 and report.incumbent.cost_per_request_usd == pytest.approx(0.00045)
    assert len(report.catalog.sha256) == 64 and report.catalog.as_of == "2026-10-24"
    assert report.thresholds.min_requests == 50


def test_each_candidate_gets_the_verdict_its_numbers_earn(catalog_path):
    report, _ = full_run(catalog_path)
    candidates = by_model(report)
    assert list(candidates) == ["good-model", "bad-model", "other-model"]  # catalog order; pricey was dropped

    good = candidates["good-model"]
    assert good.verdict.root == "safe_to_swap" and good.agreement.n == 120 and good.agreement.rate == 1.0
    assert not good.stopped_early and good.conversion.converted_requests == 0
    assert [c.category for c in good.by_category] == ["support"]  # no tags: the component is the category

    bad = candidates["bad-model"]
    assert bad.verdict.root == "do_not_swap" and bad.stopped_early and bad.requests_run == 50 and bad.agreement.rate == 0.0

    other = candidates["other-model"]  # a different provider: its requests were converted
    assert other.verdict.root == "swap_with_caveats" and other.conversion.converted_requests == 120
    assert other.conversion.unconverted[0].field.startswith("parameters.seed") and other.conversion.unconverted[0].requests == 120
    assert "converted requests" in other.explanation


def test_latency_and_cost_are_measured_per_candidate(catalog_path):
    good = by_model(full_run(catalog_path)[0])["good-model"]
    assert good.latency.p50_ms == 40.0 and good.latency.n == 120 and good.latency.failed == 0
    assert good.cost.per_request_usd == pytest.approx((100 * 0.15 + 20 * 0.6) / 1e6)
    assert good.cost.total_usd == pytest.approx(good.cost.per_request_usd * 120)


def test_dropped_candidates_and_their_reasons_are_in_the_report(catalog_path):
    report, _ = full_run(catalog_path)
    dropped = {(d.provider, d.model): d for d in report.dropped_candidates}
    assert set(dropped) == {("openai", "gpt-4o"), ("mistral", "mistral-large"), ("openai", "pricey-model")}
    assert dropped[("openai", "pricey-model")].stage == 2


def test_the_stages_are_announced_in_order(catalog_path):
    _, stages = full_run(catalog_path)
    assert stages == ["prepare", "candidates", "baseline", "candidate openai/good-model", "candidate openai/bad-model",
                      "candidate anthropic/other-model", "report"]  # fmt: skip


def test_a_stop_request_leaves_every_candidate_insufficient_but_still_writes_the_report(catalog_path):
    report, stages = full_run(catalog_path, should_cancel=lambda: True)
    assert [c.verdict.root for c in report.candidates] == ["insufficient_data"] * 3
    assert all(c.requests_run == 0 and "stopped" in c.explanation for c in report.candidates)
    assert not any(s.startswith("candidate ") for s in stages)


def test_a_stop_part_way_keeps_what_finished(catalog_path):
    state = {"stop": False}

    def stop_after_first():
        return state["stop"]

    def on_stage(stage):
        if stage == "candidate openai/bad-model":
            state["stop"] = True

    report = run(run_evaluation(records(120), catalog_path, config(cost_cap_usd=5.0), factory(), should_cancel=stop_after_first, on_stage=on_stage, now=lambda: NOW))
    verdicts = {c.model: c.verdict.root for c in report.candidates}
    assert verdicts == {"good-model": "safe_to_swap", "bad-model": "insufficient_data", "other-model": "insufficient_data"}


def test_a_spending_cap_too_small_for_the_baseline_is_a_run_error(catalog_path):
    with pytest.raises(RunError, match="spending cap"):
        full_run(catalog_path, cfg=config(spending_cap_usd=0.01))


def test_a_run_never_calls_a_provider_off_the_allow_list(catalog_path):
    calls: list = []
    full_run(catalog_path, make=factory(calls))
    assert {provider for provider, _ in calls} <= {"openai", "anthropic"}


def test_a_baseline_where_every_call_failed_is_unmeasurable_not_a_zero_noise_floor(catalog_path):
    """With a noise floor of 0, a candidate that fails just as often would look safe to swap."""

    def failing(source_provider):
        async def complete(provider, model, request):
            return ReplayResult(None, None, None, None, None, "HTTP 500")

        return complete

    report, _ = full_run(catalog_path, make=failing)
    assert report.noise_floor.value is None
    assert [c.verdict.root for c in report.candidates] == ["insufficient_data"] * 3
    assert all("noise floor" in c.explanation for c in report.candidates)
    assert not list(VALIDATOR.iter_errors(report.model_dump(mode="json", by_alias=True)))


def test_a_mostly_failing_baseline_is_unmeasurable_but_an_occasional_failure_is_not():
    cfg = config()

    def failing_when(should_fail):
        def make(source_provider):
            counter = {"i": 0}

            async def complete(provider, model, request):
                counter["i"] += 1
                if should_fail(counter["i"]):
                    return ReplayResult(None, None, None, None, None, "HTTP 500")
                return ReplayResult(f"answer to {request.messages[-1].content}", None, 100, 20, 40.0, None)

            return complete

        return make

    prepared = prepare(records(40), cfg)
    assert run(measure_baseline(prepared, None, failing_when(lambda i: i % 3 != 0), cfg)).noise_floor is None  # two in three fail
    assert run(measure_baseline(prepared, None, failing_when(lambda i: i % 10 == 0), cfg)).noise_floor is not None  # one in ten


def test_the_run_does_not_need_the_default_clock(catalog_path):
    report = run(run_evaluation(records(60), catalog_path, config(), factory()))
    assert report.generated_at.tzinfo is not None


# --- write_report -------------------------------------------------------------------------


def test_the_report_file_is_valid_json_the_schema_accepts(catalog_path, tmp_path):
    report, _ = full_run(catalog_path)
    path = tmp_path / "report.json"
    write_report(report, path)
    data = json.loads(path.read_text())
    assert not list(VALIDATOR.iter_errors(data)) and data["component"] == "support"


@pytest.mark.skipif(os.name != "posix", reason="file modes")
def test_only_the_owner_can_read_the_report_even_when_overwriting(catalog_path, tmp_path):
    report, _ = full_run(catalog_path)
    path = tmp_path / "report.json"
    path.write_text("old")
    path.chmod(0o644)
    write_report(report, path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and json.loads(path.read_text())["component"] == "support"
