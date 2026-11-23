import { useEffect, useState } from "react";
import { ApiError, cancelRun, getRun } from "./api";
import type { RunStatus } from "./api-types";
import { usd } from "./format";

export interface RunScreenProps {
  run: RunStatus;
  onViewReport: (run: RunStatus) => void; // offered once the run is done or stopped with a report
}

const ACTIVE = ["queued", "running", "cancelling"];

// On AWS Batch, the stages before the run's own: an instance booting, then the image pull. Batch
// reports no progress for these, and a warm instance takes seconds where a cold one takes minutes,
// so the screen shows the time waited rather than an estimate.
const MACHINE_STAGES = ["waiting for a machine", "starting the worker"];
const SLOW_MACHINE_SECONDS = 600; // past this, instances usually can't launch at all

function clock(seconds: number): string {
  const s = Math.floor(seconds);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function describe(stage: string): string {
  if (stage === "waiting for a machine") return "Waiting for a machine";
  if (stage === "starting the worker") return "Starting the worker";
  if (stage === "prepare") return "Reading the records";
  if (stage === "candidates") return "Choosing candidate models";
  if (stage === "baseline") return "Measuring the current model (three replays)";
  if (stage === "report") return "Writing the report";
  if (stage.startsWith("candidate ")) return `Testing ${stage.slice("candidate ".length)}`;
  return stage;
}

/**
 * Step 3: watch the run. Polls the server about once a second while the run is active and stops when
 * it is not (and when the screen goes away). Shows the finished stages, the current one, the spend so
 * far against the cap, and a Stop button.
 */
export function RunScreen({ run: initial, onViewReport }: RunScreenProps) {
  const [run, setRun] = useState(initial);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stopped = false;
    let timer: number | undefined;
    async function poll() {
      try {
        const next = await getRun(initial.id);
        if (stopped) return;
        setRun(next);
        setError(null);
        if (ACTIVE.includes(next.state)) timer = window.setTimeout(poll, 1000);
      } catch (e) {
        if (stopped) return;
        setError(e instanceof ApiError ? e.message : "Lost contact with the server.");
        timer = window.setTimeout(poll, 3000); // the server may just be busy: keep trying
      }
    }
    void poll();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [initial.id]);

  async function stop() {
    try {
      setRun(await cancelRun(run.id));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not stop the run.");
    }
  }

  const active = ACTIVE.includes(run.state);
  const stages = [...run.stages_done, ...(run.stage && !run.stages_done.includes(run.stage) ? [run.stage] : [])];
  const spend = run.spent_usd;
  const cap = run.spending_cap_usd;
  // The screen re-renders on every poll (about once a second), which keeps this current.
  const preparing = active && run.stage !== null && MACHINE_STAGES.includes(run.stage);
  const waited = Math.max(0, (Date.now() - new Date(run.started_at).getTime()) / 1000);
  const callsDone = run.calls_done;
  const callsTotal = run.calls_total;
  const counting = active && callsDone != null && callsTotal != null && callsTotal > 0;

  return (
    <main>
      <section aria-labelledby="run-h">
        <h2 id="run-h">
          3. {run.component}: {run.state === "queued" ? "waiting to start" : run.state === "running" ? "running" : run.state === "cancelling" ? "stopping" : run.state === "done" ? "finished" : run.state === "cancelled" ? "stopped" : "failed"}
        </h2>

        <div className="card">
          <ol className="stages" aria-live="polite">
            {stages.length === 0 && <li className="current">Starting…</li>}
            {stages.map((stage) => {
              const done = run.stages_done.includes(stage) || !active;
              return (
                <li key={stage} className={done ? "done" : "current"}>
                  <span aria-hidden="true">{done ? "✓" : "…"}</span> {describe(stage)}
                  {!done && counting && stage === run.stage && <span className="muted-text"> — {callsDone} of {callsTotal} requests</span>}
                  {!done && preparing && stage === run.stage && <span className="muted-text"> — {clock(waited)}</span>}
                  {!done && <span className="visually-hidden"> (in progress)</span>}
                </li>
              );
            })}
          </ol>

          {preparing && waited >= SLOW_MACHINE_SECONDS && (
            <p className="hint">
              Over 10 minutes usually means AWS can't launch an instance (instance type, quota or account limits). Check the
              compute environment's Auto Scaling activity in the EC2 console.
            </p>
          )}

          {counting && (
            <div className="meter">
              <label>
                <span>
                  Requests in this stage: {callsDone} of {callsTotal}
                </span>
                <progress value={Math.min(callsDone, callsTotal)} max={callsTotal} />
              </label>
            </div>
          )}

          {spend != null && cap != null && (
            <div className="meter">
              <label>
                <span>
                  Spent {usd(spend)} of {usd(cap)}
                </span>
                <progress value={Math.min(spend, cap)} max={cap} />
              </label>
            </div>
          )}
        </div>

        {error && <p className="error" role="alert">{error}</p>}
        {run.state === "failed" && (
          <p className="error" role="alert">
            <strong>The run failed.</strong> {run.error ?? "No reason was given."}
          </p>
        )}
        {run.state === "cancelling" && run.batch_job_id && (
          <p className="notice">Stopping: calls already sent finish first, which can take up to 30 seconds.</p>
        )}
        {run.state === "cancelled" && <p className="notice">The run was stopped. Candidates that had not finished are marked &ldquo;insufficient data&rdquo;.</p>}

        <div className="actions">
          {active && (
            <button type="button" className="btn danger" disabled={run.state === "cancelling"} onClick={() => void stop()}>
              {run.state === "cancelling" ? "Stopping…" : "Stop"}
            </button>
          )}
          {(run.state === "done" || run.state === "cancelled") && run.has_report && (
            <button type="button" className="btn primary" onClick={() => onViewReport(run)}>
              View the report
            </button>
          )}
        </div>
      </section>
    </main>
  );
}
