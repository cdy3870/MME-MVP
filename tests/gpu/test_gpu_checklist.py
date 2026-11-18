"""GPU-9: the ordered checklist, and the assembled GPU report."""

from __future__ import annotations

import pytest

from migration_eval.gpu.checklist import build_checklist
from migration_eval.gpu.labels import UNTESTED_NOTICE
from migration_eval.gpu.report import build_gpu_report

# tests/ has no __init__.py files, so pytest puts this directory on sys.path (prepend mode).
from test_gpu_assessment import profile, sample
from test_gpu_recommendations import rec



def test_fixes_first_then_runtime_then_capacity():
    items = build_checklist(
        [rec("cap", 30, 50, stage="capacity"), rec("rt", 10, 20), rec("fix", 0, 5, stage="fix")]
    )
    assert [i.source_id for i in items] == ["fix", "rt", "cap"]
    assert [i.step for i in items] == [1, 2, 3]


def test_within_a_stage_highest_gain_first():
    items = build_checklist([rec("small", 0, 5), rec("big", 20, 40)])
    assert [i.source_id for i in items] == ["big", "small"]


def test_report_carries_the_untested_notice():
    p = profile([sample(0, i, 90) for i in range(10)] + [sample(1, i, 10) for i in range(10)])
    report = build_gpu_report(p)
    assert report.untested_notice == UNTESTED_NOTICE
    assert report.costs is None  # no price table given


def test_consolidation_with_savings_becomes_the_last_capacity_step():
    from migration_eval.gpu.prices import PriceTable
    from test_gpu_prices import H100

    p = profile([sample(g, i, 30) for g in (0, 1) for i in range(10)])
    report = build_gpu_report(p, PriceTable(prices=[H100.model_copy(update={"match": ["H100"]})]))
    last = report.checklist[-1]
    assert (last.source_id, last.stage, last.expected_gain.unit) == ("consolidation", "capacity", "USD/month")


def test_gpu_report_command_writes_json(tmp_path):
    import json

    from typer.testing import CliRunner

    from migration_eval import cli
    from migration_eval.gpu.profile import write_profile

    p = profile([sample(g, i, 90 if g == 0 else 10) for g in (0, 1) for i in range(10)])
    write_profile(p, tmp_path / "gpu_profile.json")
    out = tmp_path / "gpu_report.json"
    result = CliRunner().invoke(
        cli.app,
        ["gpu", "report", str(tmp_path / "gpu_profile.json"), "--prices", str(tmp_path / "none.yaml"), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["costs"] is None and data["checklist"][0]["stage"] == "fix"
