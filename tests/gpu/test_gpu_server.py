"""The GPU report routes: upload a gpu_profile.json, read the report, list and delete."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from migration_eval.server.app import create_app

# tests/ has no __init__.py files, so pytest puts this directory on sys.path (prepend mode).
from test_gpu_assessment import profile, sample

CATALOG = Path(__file__).resolve().parents[2] / "catalog" / "catalog.yaml"


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path / "data", CATALOG))


def profile_bytes() -> bytes:
    p = profile([sample(g, i, 90 if g == 0 else 10) for g in (0, 1) for i in range(10)])
    return p.model_dump_json().encode("utf-8")


def test_upload_builds_and_stores_a_report(client: TestClient):
    created = client.post("/api/gpu/reports", files={"file": ("gpu_profile.json", profile_bytes(), "application/json")})
    assert created.status_code == 201, created.text
    info = created.json()
    assert (info["hostname"], info["gpu_count"], info["has_costs"]) == ("gpu01", 2, True)  # the repo's price file exists

    report = client.get(f"/api/gpu/reports/{info['id']}").json()
    assert report["untested_notice"]
    assert report["checklist"][0]["stage"] == "fix"  # GPUs on different sockets: pin_numa
    assert report["costs"]["missing"]  # the repo's price file has no prices yet

    assert [r["id"] for r in client.get("/api/gpu/reports").json()] == [info["id"]]
    assert client.delete(f"/api/gpu/reports/{info['id']}").status_code == 204
    assert client.get(f"/api/gpu/reports/{info['id']}").status_code == 404


def test_bad_profile_is_refused_with_the_reason(client: TestClient):
    r = client.post("/api/gpu/reports", files={"file": ("x.json", b'{"schema_version": "gpu-v9"}', "application/json")})
    assert r.status_code == 400 and "gpu-v9" in r.json()["detail"]


def test_unknown_upload_id_is_404(client: TestClient):
    r = client.post(
        "/api/gpu/reports",
        files={"file": ("gpu_profile.json", profile_bytes(), "application/json")},
        data={"upload_id": "0" * 32},
    )
    assert r.status_code == 404


def test_ids_that_are_not_ours_are_404(client: TestClient):
    assert client.get("/api/gpu/reports/..%2F..%2Fetc").status_code == 404
