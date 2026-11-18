import { useCallback, useEffect, useRef, useState } from "react";
import { formatDate } from "../format";
import { ApiError, deleteGpuReport, listGpuReports, uploadGpuProfile } from "./api";
import type { GpuReportInfo } from "./gpu-types";

export interface GpuScreenProps {
  onOpen: (info: GpuReportInfo) => void; // show the report
}

/**
 * The GPU report's entry point: drop the gpu_profile.json that `migration-eval gpu capture` (or `gpu
 * import`) wrote, and the server builds the report (GPU-5 to GPU-9). Earlier GPU reports are listed
 * below, to open again or delete.
 */
export function GpuScreen({ onOpen }: GpuScreenProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [reports, setReports] = useState<GpuReportInfo[] | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      setReports(await listGpuReports());
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not load the GPU reports.");
    }
  }, []);
  useEffect(() => void load(), [load]);

  async function send(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      onOpen(await uploadGpuProfile(file));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "The upload failed.");
    } finally {
      setBusy(false);
      if (input.current) input.current.value = ""; // so the same file can be picked again
    }
  }

  async function remove(id: string) {
    try {
      await deleteGpuReport(id);
      setConfirming(null);
      await load();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not delete the report.");
    }
  }

  return (
    <main>
      <section aria-labelledby="gpu-upload-h">
        <h2 id="gpu-upload-h">GPU report: upload a GPU profile</h2>
        <p className="section-note">
          On the GPU machine, run <code>migration-eval gpu capture</code> while it serves normal traffic, then drop the{" "}
          <code>gpu_profile.json</code> it writes here. The report shows what was measured, ranked changes to try, cost estimates and a
          checklist.
        </p>
        <div
          className={`dropzone${dragging ? " over" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            void send(e.dataTransfer.files[0]);
          }}
        >
          <p>{busy ? "Building the report…" : "Drop gpu_profile.json here"}</p>
          <button type="button" className="btn" disabled={busy} onClick={() => input.current?.click()}>
            Choose a file
          </button>
          <input ref={input} type="file" accept=".json,application/json" hidden onChange={(e) => void send(e.target.files?.[0])} />
        </div>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </section>

      <section aria-labelledby="gpu-past-h">
        <h2 id="gpu-past-h">Earlier GPU reports</h2>
        {reports === null && !error && <p className="muted-text">Loading…</p>}
        {reports?.length === 0 && <p className="empty">None yet.</p>}
        {reports && reports.length > 0 && (
          <div className="card table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">Machine</th>
                  <th scope="col">GPUs</th>
                  <th scope="col">Created</th>
                  <th scope="col" className="num">Changes</th>
                  <th scope="col"><span className="visually-hidden">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {reports.map((r) => (
                  <tr key={r.id} style={{ cursor: "default" }}>
                    <th scope="row">
                      <button type="button" className="link" onClick={() => onOpen(r)}>
                        {r.hostname}
                      </button>
                      <span className="sub">{r.filename}</span>
                    </th>
                    <td>
                      {r.gpu_count} × {r.gpu_names.join(", ")}
                    </td>
                    <td>{formatDate(r.created_at)}</td>
                    <td className="num">{r.recommendations}</td>
                    <td className="num">
                      {confirming === r.id ? (
                        <>
                          <span className="fineprint">Remove this report from the server? </span>
                          <button type="button" className="btn danger" onClick={() => void remove(r.id)}>
                            Delete
                          </button>{" "}
                          <button type="button" className="btn" onClick={() => setConfirming(null)}>
                            Keep
                          </button>
                        </>
                      ) : (
                        <button type="button" className="btn" onClick={() => setConfirming(r.id)}>
                          Delete
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </main>
  );
}
