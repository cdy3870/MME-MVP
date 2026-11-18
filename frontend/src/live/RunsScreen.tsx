import { useCallback, useEffect, useState } from "react";
import { formatDate } from "../format";
import { ApiError, deleteRun, listRuns } from "./api";
import type { RunSummary } from "./api-types";

export interface RunsScreenProps {
  onOpen: (run: RunSummary) => void; // a finished run opens its report, an unfinished one its progress
  onBack: () => void;
}

const STATE_LABEL: Record<string, string> = {
  queued: "Waiting", running: "Running", cancelling: "Stopping", done: "Done", failed: "Failed", cancelled: "Stopped",
};

/** The list of past runs: component, when it started, its state, and the verdict counts of a finished one. */
export function RunsScreen({ onOpen, onBack }: RunsScreenProps) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setRuns(await listRuns());
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not load the runs.");
    }
  }, []);
  useEffect(() => void load(), [load]);

  async function remove(id: string) {
    try {
      await deleteRun(id);
      setConfirming(null);
      await load();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not delete the run.");
    }
  }

  return (
    <main>
      <section aria-labelledby="runs-h">
        <h2 id="runs-h">Past runs</h2>
        {error && <p className="error" role="alert">{error}</p>}
        {runs === null && !error && <p className="muted-text">Loading…</p>}
        {runs?.length === 0 && <p className="empty">No runs yet. Upload an export to start one.</p>}
        {runs && runs.length > 0 && (
          <div className="card table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">Component</th>
                  <th scope="col">Started</th>
                  <th scope="col">State</th>
                  <th scope="col">Verdicts</th>
                  <th scope="col"><span className="visually-hidden">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.id} style={{ cursor: "default" }}>
                    <th scope="row">
                      <button type="button" className="link" onClick={() => onOpen(run)}>{run.component}</button>
                    </th>
                    <td>{formatDate(run.started_at)}</td>
                    <td>{STATE_LABEL[run.state] ?? run.state}</td>
                    <td>
                      {run.verdicts
                        ? Object.entries(run.verdicts).map(([verdict, n]) => `${n} ${verdict.replace(/_/g, " ")}`).join(", ") || "no candidates"
                        : "—"}
                    </td>
                    <td className="num">
                      {confirming === run.id ? (
                        <>
                          <span className="fineprint">Remove the report and its data from the server? </span>
                          <button type="button" className="btn danger" onClick={() => void remove(run.id)}>Delete</button>{" "}
                          <button type="button" className="btn" onClick={() => setConfirming(null)}>Keep</button>
                        </>
                      ) : (
                        <button type="button" className="btn" onClick={() => setConfirming(run.id)}>Delete</button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
      <p>
        <button type="button" className="link" onClick={onBack}>Back to upload</button>
      </p>
    </main>
  );
}
