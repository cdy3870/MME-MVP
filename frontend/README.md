# Report viewer (prototype)

Shows an evaluation report: the current model's baseline and noise floor, each candidate's verdict
with agreement and its 95% interval, speed and cost against the current model, what could not be
converted, and which models were never tested.

It reads reports shaped by `schema/report.v0.schema.json`; the TypeScript types are generated into
`ts/report.v0.d.ts` (run `python scripts/generate_models.py` after changing the schema). The example
reports in `schema/examples/report/valid/` are bundled, so there is no backend to run.

    npm install
    npm run dev          # http://localhost:5173
    npm run typecheck
    npm test
    npm run build

Open `?theme=light|dark` and `?report=<example name>` to link to a view. "Open a report file" loads
a report from disk; it only checks that the file looks like a v0 report (the evaluator validates
against the schema).

## Live mode: upload, run, read the report

Live mode talks to the local server instead of reading example files.

    # terminal 1: the server (keys come from ITS environment, never from the page)
    export ANTHROPIC_API_KEY=...            # and/or OPENAI_API_KEY, GEMINI_API_KEY
    migration-eval serve                    # http://127.0.0.1:8000, data in .migration-eval-data/

    # terminal 2: the UI with hot reload (proxies /api to the server)
    cd frontend && npm install && npm run dev      # http://localhost:5173/?live=1

Or build once (`npm run build`) and open http://127.0.0.1:8000/?live=1: the server serves the built page itself.

The flow: create an export with `migration-eval export`, drop the `.tar.gz` on the first screen, pick a
component, set the spending cap and the providers candidates may be called on, confirm what will be sent,
and watch the run. `?run=<id>` links straight to a run.

The server listens on this machine only and has no login. It refuses other addresses unless you pass
`--allow-remote`. Uploads and reports are stored owner-only under the data directory and can be deleted
from the screens; `.migration-eval-data/` is git-ignored because it holds your logged prompts.

Replay speed: 1x replays at the logged pace, so logs spread over days take days. The configure screen
suggests a speed that fits a pass into about ten minutes; latency is only comparable at 1x.
