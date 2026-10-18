"""Generate the Pydantic models and TypeScript types from the record schema.

schema/record.v0.schema.json is the only source of truth (REC-2). Never edit
the generated files by hand; change the schema and rerun this script.

    python scripts/generate_models.py           # regenerate both outputs
    python scripts/generate_models.py --check   # fail if outputs are stale (for CI)

Requires the dev environment: `pip install -e ".[dev]"` and `npm ci`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "schema" / "record.v0.schema.json"
PY_OUT = ROOT / "src" / "migration_eval" / "models" / "record.py"
TS_OUT = ROOT / "ts" / "record.v0.d.ts"
JSON2TS = ROOT / "node_modules" / ".bin" / "json2ts"

PY_HEADER = (
    "# DO NOT EDIT. Generated from schema/record.v0.schema.json by "
    "scripts/generate_models.py.\n"
    "# To change these models, edit the schema and rerun the script."
)


def generate_python(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "datamodel_code_generator",
            "--input",
            str(SCHEMA),
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
            PY_HEADER,
        ],
        check=True,
        cwd=ROOT,
    )


def generate_typescript(out: Path) -> None:
    if not JSON2TS.exists():
        sys.exit(f"{JSON2TS} not found. Run `npm ci` first.")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(JSON2TS), "--input", str(SCHEMA), "--output", str(out)],
        check=True,
        cwd=ROOT,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="generate to a temp dir and fail if the committed outputs differ",
    )
    args = parser.parse_args()

    if not args.check:
        generate_python(PY_OUT)
        generate_typescript(TS_OUT)
        print(f"wrote {PY_OUT.relative_to(ROOT)}")
        print(f"wrote {TS_OUT.relative_to(ROOT)}")
        return 0

    stale = []
    with tempfile.TemporaryDirectory() as tmp:
        for target, generate in ((PY_OUT, generate_python), (TS_OUT, generate_typescript)):
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
