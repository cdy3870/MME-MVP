"""Load the model catalog (CAT-1): the hand-maintained list of API offerings.

`catalog/catalog.yaml` is checked against `catalog/catalog.schema.json` through the models
generated from that schema (`models/catalog.py`). On top of what the schema can express,
the loader rejects impossible dates, repeated regions and a repeated (provider, model)
pair. Problems are reported together, naming the offering, so a pull request that breaks
the catalog says exactly what to fix. It works in two stages: the schema problems first,
then these extra checks once the schema is satisfied, so a fix may reveal more to fix.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from migration_eval.models.catalog import Catalog, Offering

import hashlib

__all__ = ["Catalog", "CatalogError", "Offering", "load_catalog"]

MAX_PROBLEMS_SHOWN = 20

# What a malformed value in each field should have looked like.
_FIELD_HINTS = {
    "as_of": "must be a date written YYYY-MM-DD",
    "source_url": "must be an https:// link",
    "regions": "must be a lowercase identifier such as us or eu",
}


class CatalogError(Exception):
    """The catalog file is unusable; the message lists every problem found."""


class _Loader(yaml.SafeLoader):
    """A SafeLoader with two guards against silent mistakes.

    * Dates stay text. By default YAML turns `2026-10-24` into a date object, so an
      unquoted `as_of` would fail a string check for no good reason.
    * A key written twice in one mapping is an error, not "the last one wins". Otherwise
      a duplicated `as_of` would quietly hide the first value.
    """

    yaml_implicit_resolvers = {
        first_char: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:timestamp"]
        for first_char, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_catalog(path: Path) -> Catalog:
    """Read and validate the catalog at `path`. Raises CatalogError listing every problem."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.load(text, Loader=_Loader)  # noqa: S506 - _Loader is a SafeLoader
    except yaml.YAMLError as exc:
        raise CatalogError(f"{path}: not valid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise CatalogError(f"{path}: expected a mapping with an `offerings` list at the top level.")

    try:
        catalog = Catalog.model_validate(data)
    except ValidationError as exc:
        raise CatalogError(_report(path, _schema_problems(exc, data))) from None

    problems = _semantic_problems(catalog)
    if problems:
        raise CatalogError(_report(path, problems))
    return catalog


# --- problems --------------------------------------------------------------------------------


def _report(path: Path, problems: list[str]) -> str:
    noun = "problem" if len(problems) == 1 else "problems"
    lines = [f"{path} has {len(problems)} {noun}:"]
    lines.extend(f"  - {p}" for p in problems[:MAX_PROBLEMS_SHOWN])
    if len(problems) > MAX_PROBLEMS_SHOWN:
        lines.append(f"  ... and {len(problems) - MAX_PROBLEMS_SHOWN} more")
    return "\n".join(lines)


def _label(index: int, provider: object, model: object) -> str:
    return f"offerings[{index}] ({provider}/{model})"


def _schema_problems(exc: ValidationError, data: dict[str, Any]) -> list[str]:
    problems = []
    for error in exc.errors():
        loc = error["loc"]
        # The field a hint applies to may be above the leaf (regions.0 is inside regions).
        hint_field = next((str(part) for part in loc if str(part) in _FIELD_HINTS), None)

        # A short phrase that continues the sentence ("as_of is missing"), or else
        # Pydantic's own message, which follows a colon ("context_window: Input should ...").
        phrase = message = None
        if error["type"] == "missing":
            phrase = "is missing"
        elif error["type"] == "extra_forbidden":
            phrase = "is not a known field"
        elif error["type"] in ("string_pattern_mismatch", "string_type") and hint_field:
            phrase = f"{_FIELD_HINTS[hint_field]} (got {error['input']!r})"
        else:
            message = error["msg"]

        if len(loc) >= 2 and loc[0] == "offerings" and isinstance(loc[1], int):
            offering = _offering_at(data, loc[1])
            where = _label(loc[1], offering.get("provider", "?"), offering.get("model", "?"))
            field = ".".join(str(part) for part in loc[2:])
            subject = f"{where}: {field}" if field else where
        else:
            subject = ".".join(str(part) for part in loc) or "catalog"

        problems.append(f"{subject} {phrase}" if phrase else f"{subject}: {message}")
    return problems


def _offering_at(data: dict[str, Any], index: int) -> dict[str, Any]:
    offerings = data.get("offerings")
    if isinstance(offerings, list) and index < len(offerings) and isinstance(offerings[index], dict):
        return offerings[index]
    return {}


def _semantic_problems(catalog: Catalog) -> list[str]:
    """Checks the generated models do not enforce.

    That is real calendar dates, unique regions within an offering (the schema says
    `uniqueItems` but the generated model drops it), and unique (provider, model) pairs
    across the catalog, which JSON Schema cannot express at all.
    """
    problems = []
    seen: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, offering in enumerate(catalog.offerings):
        where = _label(index, offering.provider, offering.model)
        seen[(offering.provider, offering.model)].append(index)
        try:
            date.fromisoformat(offering.as_of)
        except ValueError:
            problems.append(f"{where}: as_of {offering.as_of!r} is not a real date")
        repeated = sorted(r for r, n in Counter(offering.regions).items() if n > 1)
        if repeated:
            problems.append(f"{where}: regions lists {', '.join(map(repr, repeated))} more than once")
    for (provider, model), indexes in seen.items():
        if len(indexes) > 1:
            positions = ", ".join(f"offerings[{i}]" for i in indexes)
            problems.append(f"{provider}/{model} is listed more than once ({positions})")
    return problems


# --- what a report records about the catalog it used (CAT-6) -------------------------------


def catalog_hash(path: Path) -> str:
    """The SHA-256 (hex) of the catalog file's raw bytes.

    Hash the bytes exactly as stored, with no parsing or reformatting, so the same file always
    gives the same hash and any edit, even to a comment, changes it. Reading the file raises
    the normal FileNotFoundError when it is missing.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def catalog_as_of(catalog: Catalog) -> date:
    """The catalog's as-of date for a report: the OLDEST `as_of` among its offerings.

    A report is only as fresh as its stalest figure, so the oldest date is the honest one.
    (The alternative is the newest; say so here if you change it.) `as_of` is text, so parse
    it with `date.fromisoformat`. A loaded catalog always has at least one offering.
    """

    return min(date.fromisoformat(o.as_of) for o in catalog.offerings)
    


def catalog_provenance(path: Path) -> dict[str, str]:
    """What every report records about its catalog (CAT-6, and NFR-11's reproducibility):

        {"catalog_sha256": "<hex>", "catalog_as_of": "YYYY-MM-DD"}

    Load the catalog at `path` (raising CatalogError if it is invalid), then combine
    `catalog_hash` and `catalog_as_of`.
    """

    catalog = load_catalog(path)

    return {
        "catalog_sha256": catalog_hash(path),
        "catalog_as_of": catalog_as_of(catalog).isoformat(),
    }
