"""Request counts while a run is going: `run_evaluation(on_progress=...)` and `replay(on_request_done=...)`."""

from __future__ import annotations

import asyncio

from migration_eval.noise_floor import NOISE_FLOOR_RUNS
from migration_eval.replay import replay
from migration_eval.run import run_evaluation

# tests/ has no __init__.py files, so pytest puts this directory on sys.path (prepend mode).
from test_run import catalog_path, config, factory, records  # noqa: F401  (catalog_path is a fixture)


def test_replay_reports_each_finished_request_once():
    done: list[int] = []
    sample = records(5)
    complete = factory()("openai")
    asyncio.run(replay(sample, "openai", "gpt-4o", complete, speed=1e6, on_request_done=lambda: done.append(1)))
    assert len(done) == 5


def test_run_evaluation_counts_requests_per_stage(catalog_path):  # noqa: F811
    events: list[tuple[str, int, int]] = []
    stage = {"name": ""}

    def on_stage(name: str) -> None:
        stage["name"] = name

    def on_progress(done: int, total: int) -> None:
        events.append((stage["name"], done, total))

    n = 60
    asyncio.run(run_evaluation(records(n), catalog_path, config(), factory(), on_stage=on_stage, on_progress=on_progress))

    baseline = [(d, t) for s, d, t in events if s == "baseline"]
    assert baseline[0] == (0, NOISE_FLOOR_RUNS * n) and baseline[-1] == (NOISE_FLOOR_RUNS * n, NOISE_FLOOR_RUNS * n)
    assert [d for d, _ in baseline] == list(range(NOISE_FLOOR_RUNS * n + 1))

    candidates = sorted({s for s, _, _ in events if s.startswith("candidate ")})
    assert candidates, "the test catalog has candidates on the allowed providers"
    for name in candidates:
        counts = [(d, t) for s, d, t in events if s == name]
        assert counts[0] == (0, n)
        assert all(t == n for _, t in counts) and [d for d, _ in counts] == list(range(len(counts)))
        assert counts[-1][0] <= n  # early stopping can end a candidate before the whole sample

    assert not [e for e in events if e[0] in ("prepare", "candidates", "report")], "only replaying stages count"
