"""GPU-10: gpu_prices.yaml."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from migration_eval.capture.file_checks import FileRejected
from migration_eval.gpu.prices import DEFAULT_PRICES_PATH, GpuPrice, PriceTable, load_prices, match_gpu


H100 = GpuPrice(
    gpu="H100 80GB", match=["H100", "80GB"], hourly_usd_low=2.0, hourly_usd_high=4.0,
    memory_gib=80, as_of=date(2026, 11, 17), source_url="https://example.com",
)  # fmt: skip


def test_repo_price_file_loads():
    assert isinstance(load_prices(DEFAULT_PRICES_PATH), PriceTable)


def test_reversed_price_range_is_rejected(tmp_path: Path):
    path = tmp_path / "p.yaml"
    path.write_text(
        "prices:\n  - {gpu: X, match: [X], hourly_usd_low: 5, hourly_usd_high: 1, memory_gib: 1,"
        " as_of: '2026-11-17', source_url: 'https://example.com'}\n",
        encoding="utf-8",
    )
    with pytest.raises(FileRejected):
        load_prices(path)


def test_match_gpu_by_substrings():
    table = PriceTable(prices=[H100])
    assert match_gpu("NVIDIA H100 80GB HBM3", table) is H100
    assert match_gpu("NVIDIA GeForce GTX 1080 Ti", table) is None
