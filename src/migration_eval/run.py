"""Run an evaluation (RUN-1): the glue that chains the pieces into one report.

    records for a component
      -> profile and sample                              (profile.py, replay.select_sample)
      -> candidates: Stage 1 and Stage 2 filters         (candidates.py)
      -> the incumbent replayed 3 times -> noise floor   (noise_floor.py)
      -> each candidate on the sample, stopping early    (early_sample.py)
      -> per request: deterministic checks, then compare (checks.py, judge.py later)
      -> agreement overall and by category               (agreement.py)
      -> latency, cost, conversion flags                 (metrics.py, request_builders.py)
      -> a verdict                                       (verdict.py)
      -> a Report                                        (models/report.py, schema/report.v0.schema.json)

Nothing here calls a provider directly. The model-calling function comes from the caller, as in
the replay (`CompleteFactory` below), and the harness never calls `migration_eval.start()`: it
would record the replays.

Open decisions (see the notes on RUN-1 and RUN-4):
    * the sample size per tier. EVL-2 says 50 to 200 requests per candidate "(smaller in smoke
      runs)", and NFR-8 mentions a 200-request smoke run in under 30 minutes. They disagree; the
      numbers in `TIER_SAMPLE_SIZE` are placeholders.
    * `replay_incumbent` takes one `complete` for every run, but the cache needs a different
      replica number for each of the three runs (see cache.py), so it needs a small change: take
      a function from the run number to a `complete`.
    * `count_agreements` returns only a count, and the report needs a yes/no per request (for the
      categories and the check failures), so `request_outcomes` below replaces it for the report.
    * a `Comparator` receives two results and not the request, but the deterministic checks need
      the request, so they are applied in `request_outcomes`, before the comparator.
    * until the judge (EVL-5) exists, the comparator is `same_output` (identical text), which only
      suits structured components.
    * the stages become resumable jobs under RUN-4, so this function should be a straight line of
      stage functions that can each be run, saved and skipped on resume
    * what `spending_cap_usd` covers (candidate calls, incumbent replays, the judge) and what a
      run does when the cap is reached: stop and mark the unfinished candidates "insufficient
      data" (RUN-5, RUN-6)
    * early stopping (`run_candidate`) counts agreement with the comparator alone, while the
      report's outcomes also apply the deterministic checks, so a candidate can be stopped
      slightly later or earlier than its final numbers suggest
    * an incumbent that is not in the catalog has no prices, so its replays are not metered
    * no rate limiter is passed to the replays yet: where the partner's limits come from is open
    * a baseline where most calls failed (a bad key, an outage) is treated as unmeasurable: with
      failures counted as disagreement its noise floor would be near 0, and a candidate that fails
      just as often would then look "safe to swap". `MAX_BASELINE_FAILURE_RATE` is a placeholder.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from migration_eval.agreement import agreement, agreement_by_category
from migration_eval.candidates import API_HOSTING, Constraints, dropped_for_report, stage_one, stage_two
from migration_eval.capture.grading import Grade, grade
from migration_eval.catalog import CatalogError, Offering, catalog_provenance, load_catalog
from migration_eval.checks import CheckResult, first_failure, run_checks
from migration_eval.early_sample import CandidateRun, run_candidate
from migration_eval.metrics import LatencySummary, candidate_metrics, latency_summary
from migration_eval.models import CallRecord, Report
from migration_eval.noise_floor import Comparator, noise_floor, replay_incumbent, same_output
from migration_eval.pricing import request_cost
from migration_eval.profile import Incumbent, Profile, build_profile, records_for_component
from migration_eval.replay import CompleteFn, ReplayResult, error_result, select_sample
from migration_eval.request_builders import unconverted_for_report
from migration_eval.verdict import Thresholds, Verdict, assign_verdict, explain_verdict

Tier = Literal["smoke", "full"]

TIER_SAMPLE_SIZE: dict[Tier, int] = {"smoke": 50, "full": 200}  # placeholders, see the open decisions
MAX_BASELINE_FAILURE_RATE = 0.5  # above this share of failed incumbent calls there is no usable noise floor

# Given the provider the traffic was logged from, return the function that calls models.
# (`request_builders.make_complete` is the real one; tests pass a fake.)
CompleteFactory = Callable[[str], CompleteFn]


class RunError(Exception):
    """The run cannot start or continue for a reason the user can fix: no replay-grade records for
    the component, no incumbent, a bad catalog, an empty allow-list. The message says which."""


@dataclass(frozen=True)
class RunConfig:
    component: str
    tier: Tier  # "smoke" or "full": sets the sample size
    spending_cap_usd: float  # the most this run may spend on provider calls (RUN-5)
    allow_list: frozenset[str]  # providers candidates may be called on; empty drops everything
    cost_cap_usd: float | None = None  # Stage 2: drop candidates predicted above this per month
    hosting: str = API_HOSTING
    region: str | None = None
    window: timedelta = timedelta(days=7)  # the period the records cover, for the monthly prediction
    replay_speed: float = 1.0  # 1 is the logged pace, and the only speed at which latency is comparable
    seed: int = 1  # the same seed picks the same sample, so candidates are compared on the same requests


@dataclass(frozen=True)
class Prepared:
    """What the run works on, decided once and shared by every candidate."""

    records: list[CallRecord]  # the component's replay-grade records
    profile: Profile
    incumbent: Incumbent
    sample: list[CallRecord]  # the same requests for every candidate, in time order


@dataclass(frozen=True)
class Baseline:
    """The incumbent measured now (EVL-1, EVL-7): what candidates are compared with."""

    runs: list[list[ReplayResult]]  # one list of results per replay, each lined up with the sample
    noise_floor: float | None
    latency: LatencySummary
    cost_per_request_usd: float | None


@dataclass(frozen=True)
class CandidateEvaluation:
    """Everything the report needs about one candidate, before it is turned into the schema."""

    offering: Offering
    run: CandidateRun
    outcomes: list[bool]  # one per request run: did it agree with the incumbent?
    checks: list[list[CheckResult]]  # the deterministic checks for each request run
    cancelled: bool = False  # not finished because the run was stopped or hit its cap (RUN-6)


@dataclass
class SpendMeter:
    """Keeps the running total of what the run has spent and refuses calls that could pass the cap (RUN-5)."""

    cap_usd: float
    spent_usd: float = 0.0
    refused: int = 0  # calls not made because they might have passed the cap

    def can_afford(self, worst_case_usd: float) -> bool:
        """True when spending `worst_case_usd` more stays within the cap."""
        return self.spent_usd + worst_case_usd <= self.cap_usd

    def record(self, actual_usd: float) -> None:
        """Add what a finished call really cost (a negative amount takes back an over-estimate)."""
        self.spent_usd += actual_usd


CHARS_PER_TOKEN = 4  # a rough rule, used only to bound a request's input cost
DEFAULT_OUTPUT_TOKENS = 4096  # assumed when a request sets no output limit


def metered(complete: CompleteFn, meter: SpendMeter, offering: Offering) -> CompleteFn:
    """Wrap `complete` so each call is checked against the cap first, using the worst case cost
    (the request's input tokens plus the output limit, at `offering`'s prices), and recorded after.
    A call that would pass the cap is not sent: it returns a ReplayResult with an `error`.
    """

    async def wrapper(provider: str, model: str, request: Any) -> ReplayResult:
        input_tokens = len(request.model_dump_json()) // CHARS_PER_TOKEN + 1
        limit = request.parameters.max_output_tokens if request.parameters else None
        worst_case = request_cost(offering, input_tokens, limit or DEFAULT_OUTPUT_TOKENS)

        if not meter.can_afford(worst_case):
            meter.refused += 1
            return error_result("spending cap reached")

        meter.record(worst_case)  # reserve it first: calls run concurrently
        try:
            result = await complete(provider, model, request)
        except BaseException:
            meter.record(-worst_case)  # nothing was spent (for example a 429 that will be retried)
            raise

        if result.input_tokens is not None and result.output_tokens is not None:
            meter.record(request_cost(offering, result.input_tokens, result.output_tokens) - worst_case)
        return result

    return wrapper


# --- the stages ----------------------------------------------------------------------------


def prepare(records: list[CallRecord], config: RunConfig) -> Prepared:
    """Pick the work: the component's replay-grade records (`grading`), its profile and incumbent
    (`profile.build_profile`), and the sample (`replay.select_sample`, sized by the tier and seeded
    by `config.seed`). Raise RunError when there is nothing to evaluate.
    """
    component_records = [
        r for r in records_for_component(records, config.component) if grade(r).grade == Grade.REPLAY_GRADE
    ]
    if not component_records:
        raise RunError(f"no replay-grade records for component {config.component!r}")

    profile = build_profile(component_records, config.component)
    if profile.incumbent is None:
        raise RunError(f"could not tell which model component {config.component!r} runs on")

    sample = select_sample(component_records, TIER_SAMPLE_SIZE[config.tier], config.seed)
    return Prepared(records=component_records, profile=profile, incumbent=profile.incumbent, sample=sample)


def choose_candidates(
    catalog_path: Path, prepared: Prepared, config: RunConfig
) -> tuple[list[Offering], list[dict]]:
    """Load the catalog, run Stage 1 then Stage 2 (when a cost cap is set), and return
    `(candidates, dropped)`, where `dropped` is `dropped_for_report`'s output. An empty
    allow-list drops everything (a RunError is kinder than an empty report, say which you chose).

    An empty allow-list is a RunError here: it would drop every offering, and a report that says
    "nothing was tested" for a setup mistake is worse than telling the user.
    """
    if not config.allow_list:
        raise RunError("the provider allow-list is empty, so no candidate may be called")

    try:
        catalog = load_catalog(catalog_path)
    except CatalogError as exc:
        raise RunError(str(exc)) from exc

    constraints = Constraints(allow_list=config.allow_list, hosting=config.hosting, region=config.region)
    kept, dropped_one = stage_one(catalog, prepared.profile, constraints)

    dropped_two: list[tuple[Offering, list[str]]] = []
    if config.cost_cap_usd is not None:
        kept, dropped_two = stage_two(kept, prepared.records, config.window, config.cost_cap_usd)

    return kept, dropped_for_report(dropped_one, dropped_two)


def incumbent_offering(catalog_path: Path, incumbent: Incumbent) -> Offering | None:
    """The incumbent's catalog entry (for its prices), or None when the catalog does not list it."""
    try:
        catalog = load_catalog(catalog_path)
    except CatalogError as exc:
        raise RunError(str(exc)) from exc
    return next((o for o in catalog.offerings if (o.provider, o.model) == (incumbent.provider, incumbent.model)), None)


async def measure_baseline(
    prepared: Prepared,
    incumbent_offering: Offering | None,
    make_complete: CompleteFactory,
    config: RunConfig,
    comparator: Comparator = same_output,
    meter: SpendMeter | None = None,
) -> Baseline:
    """Replay the incumbent three times on the sample, then take the noise floor, the latency and
    the cost of those replays. `incumbent_offering` is the incumbent's catalog entry for its
    prices, or None when it is not in the catalog (cost is then None).
    """
    incumbent = prepared.incumbent
    complete = make_complete(incumbent.provider)  # the factory is told where the traffic came from
    if meter is not None and incumbent_offering is not None:
        complete = metered(complete, meter, incumbent_offering)

    runs = await replay_incumbent(
        prepared.sample, incumbent.provider, incumbent.model, complete, speed=config.replay_speed
    )

    every_call = [result for run in runs for result in run]
    if incumbent_offering is None:
        latency, cost_per_request = latency_summary(every_call), None
    else:
        measured = candidate_metrics(incumbent_offering, every_call)
        latency, cost_per_request = measured.latency, measured.cost_per_request_usd

    failed = sum(1 for result in every_call if result.error is not None)
    usable = not every_call or failed / len(every_call) <= MAX_BASELINE_FAILURE_RATE
    floor = noise_floor(runs, comparator) if usable else None

    return Baseline(runs=runs, noise_floor=floor, latency=latency, cost_per_request_usd=cost_per_request)


def request_outcomes(
    sample: list[CallRecord],
    candidate_results: list[ReplayResult],
    baseline: Baseline,
    comparator: Comparator = same_output,
) -> tuple[list[bool], list[list[CheckResult]]]:
    """For each request the candidate ran: run the deterministic checks (`checks.run_checks`
    against the incumbent's first run); a failed check means "different" without asking the
    comparator; otherwise the candidate agrees when most of the incumbent's runs agree with it
    (the same rule as `early_sample.count_agreements`). Return the yes/no list and each request's
    checks.
    """
    outcomes: list[bool] = []
    all_checks: list[list[CheckResult]] = []

    for i, (record, candidate) in enumerate(zip(sample, candidate_results)):
        incumbent_results = [run[i] for run in baseline.runs]
        checks = run_checks(record.request, incumbent_results[0], candidate)
        all_checks.append(checks)

        if first_failure(checks) is not None:
            outcomes.append(False)
            continue

        votes = sum(1 for incumbent in incumbent_results if comparator(candidate, incumbent))
        outcomes.append(votes * 2 > len(incumbent_results))

    return outcomes, all_checks


async def evaluate_candidate(
    offering: Offering,
    prepared: Prepared,
    baseline: Baseline,
    make_complete: CompleteFactory,
    config: RunConfig,
    comparator: Comparator = same_output,
    should_cancel: Callable[[], bool] = lambda: False,
    meter: SpendMeter | None = None,
) -> CandidateEvaluation:
    """Run one candidate with `early_sample.run_candidate` and turn its results into outcomes.
    Stop when `should_cancel()` is true or the spend meter refuses, and return what finished,
    marked `cancelled`.
    """
    send = make_complete(prepared.incumbent.provider)  # the factory is told where the traffic came from
    if meter is not None:
        send = metered(send, meter, offering)

    async def complete(provider: str, model: str, request: Any) -> ReplayResult:
        if should_cancel():
            return error_result("run stopped")
        return await send(provider, model, request)

    refused_before = meter.refused if meter is not None else 0
    floor = baseline.noise_floor if baseline.noise_floor is not None else 0.0  # 0 never stops a run early

    candidate_run = await run_candidate(
        offering.provider,
        offering.model,
        prepared.sample,
        baseline.runs,
        floor,
        complete,
        comparator,
        speed=config.replay_speed,
    )
    outcomes, checks = request_outcomes(prepared.sample, candidate_run.results, baseline, comparator)

    cancelled = should_cancel() or (meter is not None and meter.refused > refused_before)
    return CandidateEvaluation(offering=offering, run=candidate_run, outcomes=outcomes, checks=checks, cancelled=cancelled)


def _unfinished(offering: Offering) -> CandidateEvaluation:
    """A candidate that was never started because the run was stopped first."""
    run = CandidateRun(offering.provider, offering.model, [], None, 0, False)
    return CandidateEvaluation(offering=offering, run=run, outcomes=[], checks=[], cancelled=True)


def _candidate_entry(
    evaluation: CandidateEvaluation, prepared: Prepared, baseline: Baseline, thresholds: Thresholds
) -> dict[str, Any]:
    run, offering = evaluation.run, evaluation.offering
    ran = prepared.sample[: len(evaluation.outcomes)]

    overall = agreement(evaluation.outcomes)
    by_category = {
        name: a for name, a in agreement_by_category(ran, evaluation.outcomes).items()
    }
    metrics = candidate_metrics(offering, run.results)
    conversion = unconverted_for_report(run.results)
    converted = conversion["converted_requests"] > 0

    if evaluation.cancelled:
        verdict, explanation = Verdict.INSUFFICIENT_DATA, "The run was stopped before this candidate finished."
    elif baseline.noise_floor is None:
        verdict = Verdict.INSUFFICIENT_DATA
        explanation = "The current model could not be measured against itself, so there is no noise floor to compare with."
    else:
        verdict = assign_verdict(run, overall, by_category, baseline.noise_floor, thresholds, used_converted_requests=converted)
        explanation = explain_verdict(
            verdict, run, overall, baseline.noise_floor, thresholds, by_category=by_category, used_converted_requests=converted
        )

    failures = Counter(f.check for f in (first_failure(c) for c in evaluation.checks) if f is not None)
    return {
        "provider": offering.provider,
        "model": offering.model,
        "verdict": verdict.name.lower(),
        "explanation": explanation,
        "stopped_early": run.stopped_early,
        "requests_run": run.requests_run,
        "agreement": asdict(overall),
        "by_category": [{"category": name, "agreement": asdict(a)} for name, a in by_category.items()],
        "latency": asdict(metrics.latency),
        "cost": {"total_usd": metrics.cost_usd, "per_request_usd": metrics.cost_per_request_usd},
        "conversion": conversion,
        "check_failures": [{"check": name, "requests": n} for name, n in sorted(failures.items())],
    }


def build_report(
    config: RunConfig,
    prepared: Prepared,
    baseline: Baseline,
    evaluations: list[CandidateEvaluation],
    dropped: list[dict],
    catalog_path: Path,
    incumbent_offering: Offering | None,
    now: datetime,
) -> Report:
    """Assemble the `Report` (schema/report.v0.schema.json): per candidate the agreement overall
    and by category, latency, cost, conversion summary (`request_builders.unconverted_for_report`),
    check-failure counts, and the verdict and its explanation (`verdict.assign_verdict`,
    `explain_verdict`). A cancelled candidate is "insufficient data". Validate by building the
    model, so a schema mismatch is an error here and not in the viewer.
    """
    thresholds = Thresholds()
    provenance = catalog_provenance(catalog_path)

    return Report.model_validate(
        {
            "schema_version": "v0",
            "generated_at": now,
            "component": config.component,
            "catalog": {"sha256": provenance["catalog_sha256"], "as_of": provenance["catalog_as_of"]},
            "incumbent": {
                "provider": prepared.incumbent.provider,
                "model": prepared.incumbent.model,
                "latency": asdict(baseline.latency),
                "cost_per_request_usd": baseline.cost_per_request_usd,
            },
            "sample": {
                "records_available": len(prepared.records),
                "requests_replayed": len(prepared.sample),
                "replay_speed": config.replay_speed,
                "seed": config.seed,
            },
            "noise_floor": {"value": baseline.noise_floor, "runs": len(baseline.runs), "requests": len(prepared.sample)},
            "thresholds": asdict(thresholds),
            "candidates": [_candidate_entry(e, prepared, baseline, thresholds) for e in evaluations],
            "dropped_candidates": dropped,
        }
    )


async def run_evaluation(
    records: list[CallRecord],
    catalog_path: Path,
    config: RunConfig,
    make_complete: CompleteFactory,
    *,
    comparator: Comparator = same_output,
    on_stage: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] = lambda: False,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Report:
    """Evaluate every candidate for one component and return the report.

    The stages, in order, each announced to `on_stage` (for progress and for RUN-4's jobs):
    "prepare", "candidates", "baseline", one "candidate <provider>/<model>" per candidate, then
    "report". The candidates are run one after another (they share the rate limits), each against
    the same sample and the same incumbent runs. A stop request or the spending cap ends the run
    early and the report still gets written, with the unfinished candidates marked
    "insufficient data" (RUN-6).
    """
    announce = on_stage or (lambda stage: None)
    meter = SpendMeter(config.spending_cap_usd)

    announce("prepare")
    prepared = prepare(records, config)

    announce("candidates")
    candidates, dropped = choose_candidates(catalog_path, prepared, config)

    announce("baseline")
    incumbent_prices = incumbent_offering(catalog_path, prepared.incumbent)
    baseline = await measure_baseline(prepared, incumbent_prices, make_complete, config, comparator, meter)
    if meter.refused:
        raise RunError("the spending cap was reached before the current model's baseline finished")

    evaluations: list[CandidateEvaluation] = []
    for offering in candidates:
        if should_cancel() or any(e.cancelled for e in evaluations):
            evaluations.append(_unfinished(offering))
            continue
        announce(f"candidate {offering.provider}/{offering.model}")
        evaluations.append(
            await evaluate_candidate(offering, prepared, baseline, make_complete, config, comparator, should_cancel, meter)
        )

    announce("report")
    return build_report(config, prepared, baseline, evaluations, dropped, catalog_path, incumbent_prices, now())


def write_report(report: Report, path: Path) -> None:
    """Write the report as JSON (RPT-1's `report.json`): `model_dump(mode="json", by_alias=True)`,
    indented, with the file readable by its owner only (it describes partner traffic).
    """
    text = json.dumps(report.model_dump(mode="json", by_alias=True), indent=2) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)  # the mode above only applies when the file is new
        handle.write(text)
