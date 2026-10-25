"""Generate the Pydantic models (and TypeScript types) from the JSON schemas.

Each schema is the only source of truth for its models (REC-2, CAT-1). Never edit the
generated files by hand; change the schema and rerun this script.

    python scripts/generate_models.py           # regenerate every output
    python scripts/generate_models.py --check   # fail if outputs are stale (for CI)

Requires the dev environment: `pip install -e ".[dev]"` and `npm ci`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "src" / "migration_eval" / "models"
JSON2TS = ROOT / "node_modules" / ".bin" / "json2ts"

RECORD_SCHEMA = ROOT / "schema" / "record.v0.schema.json"
TS_OUT = ROOT / "ts" / "record.v0.d.ts"


@dataclass(frozen=True)
class PythonTarget:
    schema: Path
    out: Path
    # Extra datamodel-codegen flags for this schema only, so one schema's needs do not
    # change another's generated file.
    extra_args: tuple[str, ...] = field(default_factory=tuple)


PYTHON_TARGETS = (
    PythonTarget(RECORD_SCHEMA, MODELS / "record.py"),
    # Type aliases make constrained strings (like Region) plain `str` values instead of
    # wrapper objects, so `"us" in offering.regions` works.
    PythonTarget(
        ROOT / "catalog" / "catalog.schema.json",
        MODELS / "catalog.py",
        extra_args=("--use-type-alias",),
    ),
)


def py_header(schema: Path) -> str:
    return (
        f"# DO NOT EDIT. Generated from {schema.relative_to(ROOT).as_posix()} by "
        "scripts/generate_models.py.\n"
        "# To change these models, edit the schema and rerun the script."
    )


def generate_python(target: PythonTarget, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "datamodel_code_generator",
            "--input",
            str(target.schema),
            "--input-file-type",
            "jsonschema",
            "--output",
            str(out),
            "--output-model-type",
            "pydantic_v2.BaseModel",
            "--target-python-version",
            "3.12",
            "--use-standard-collections",
            "--use-union-operator",
            "--use-annotated",
            "--enum-field-as-literal",
            "all",
            "--extra-fields",
            "forbid",
            "--formatters",
            "black",
            "isort",
            "--custom-file-header",
            py_header(target.schema),
            *target.extra_args,
        ],
        check=True,
        cwd=ROOT,
    )


def generate_typescript(out: Path) -> None:
    if not JSON2TS.exists():
        sys.exit(f"{JSON2TS} not found. Run `npm ci` first.")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(JSON2TS), "--input", str(RECORD_SCHEMA), "--output", str(out)],
        check=True,
        cwd=ROOT,
    )


def outputs() -> list[tuple[Path, Callable[[Path], None]]]:
    """Every generated file, paired with a function that writes it to a given path."""
    jobs: list[tuple[Path, Callable[[Path], None]]] = [
        (t.out, lambda path, t=t: generate_python(t, path)) for t in PYTHON_TARGETS
    ]
    jobs.append((TS_OUT, generate_typescript))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="generate to a temp dir and fail if the committed outputs differ",
    )
    args = parser.parse_args()

    if not args.check:
        for target, generate in outputs():
            generate(target)
            print(f"wrote {target.relative_to(ROOT)}")
        return 0

    stale = []
    with tempfile.TemporaryDirectory() as tmp:
        for target, generate in outputs():
            fresh = Path(tmp) / target.name
            generate(fresh)
            if not target.exists() or target.read_bytes() != fresh.read_bytes():
                stale.append(target.relative_to(ROOT))

    if stale:
        names = ", ".join(str(p) for p in stale)
        print(f"stale generated files: {names}", file=sys.stderr)
        print("run `python scripts/generate_models.py` and commit the result", file=sys.stderr)
        return 1
    print("generated files are up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
