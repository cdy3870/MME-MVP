"""GPU-10: the hand-maintained GPU price list, `catalog/gpu_prices.yaml`.

Hourly prices per GPU type, used by the cost estimates (GPU-7). Hand-maintained and reviewed
in pull requests, like catalog.yaml. Each entry has a low and a high hourly price (e.g.
reserved vs on-demand, or cheapest vs most expensive provider), so costs come out as ranges.

Matching: nvidia-smi names GPUs like "NVIDIA H100 80GB HBM3". Each entry lists `match`
substrings ("H100 80GB"); the first entry whose substrings all appear in the name wins.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from migration_eval.capture.file_checks import FileRejected

DEFAULT_PRICES_PATH = Path("catalog/gpu_prices.yaml")


class GpuPrice(BaseModel):
    gpu: str  # display name, "H100 80GB SXM"
    match: list[str]  # substrings that must all appear in the nvidia-smi name
    hourly_usd_low: float  # per GPU per hour
    hourly_usd_high: float
    memory_gib: int
    as_of: date  # the day the figures were checked
    source_url: str


class PriceTable(BaseModel):
    prices: list[GpuPrice]


def load_prices(path: Path = DEFAULT_PRICES_PATH) -> PriceTable:
    """Read and validate the YAML. Raise FileRejected (capture.file_checks) on a missing file,
    bad YAML, an entry that doesn't validate, or hourly_usd_low > hourly_usd_high."""
    try:
        with open(path, encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileRejected(f"{path}: price file not found.") from None
    except yaml.YAMLError as exc:
        raise FileRejected(f"{path}: not valid YAML ({exc}).") from None

    try:
        table = PriceTable.model_validate(data)
    except ValidationError as exc:
        raise FileRejected(f"{path}: not a valid price file ({exc.error_count()} problems): {exc}") from None

    for price in table.prices:
        if price.hourly_usd_low > price.hourly_usd_high:
            raise FileRejected(
                f"{path}: {price.gpu}: hourly_usd_low ({price.hourly_usd_low}) is above hourly_usd_high ({price.hourly_usd_high})."
            )
    return table


def match_gpu(name: str, table: PriceTable) -> GpuPrice | None:
    """The first entry whose `match` substrings all appear in `name` (case-insensitive); None if none do."""
    name = name.lower()
    for price in table.prices:
        if all(match.lower() in name for match in price.match):
            return price
    return None
